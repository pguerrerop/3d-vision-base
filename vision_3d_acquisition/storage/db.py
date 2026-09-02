"""SQLite catalog connection and schema migrations.

Stage 0 of the catalog migration. Nothing in the API reads from this database
yet: it is built next to the filesystem so the projection can be checked
against ``get_take_summary`` before any endpoint depends on it.

Concurrency rules, in the order they matter with four writer processes (API,
25D worker, RGB worker, fusion publisher):

* one connection per process, thread-local, because FastAPI runs sync handlers
  in a threadpool and a ``sqlite3.Connection`` is not shared across threads;
* every write inside :func:`transaction`, which opens ``BEGIN IMMEDIATE`` so a
  conflict fails fast instead of halfway through;
* no write transaction stays open while a pipeline runs — open it when the run
  finishes, write the row, close it.

The applied schema version lives in ``PRAGMA user_version``, which is set inside
the same transaction that applies the migration and therefore cannot drift from
the schema it describes. ``index_meta`` holds the softer bookkeeping (last scan
timestamps, projection version).
"""

from __future__ import annotations

import re
import sqlite3
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no flock
    fcntl = None  # type: ignore[assignment]

DB_FILENAME = "index.db"
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_MIGRATION_PATTERN = re.compile(r"^(\d{3,})_[a-z0-9_]+\.sql$")

_local = threading.local()


def database_path(data_dir: Path) -> Path:
    return Path(data_dir) / DB_FILENAME


def connect(data_dir: Path, *, timeout: float = 10.0) -> sqlite3.Connection:
    """Open a catalog connection with the pragmas the concurrency model needs."""
    path = database_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=timeout, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # busy_timeout first: the pragmas below must be able to wait for a lock.
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    _ensure_wal(conn, timeout=timeout)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_wal(conn: sqlite3.Connection, *, timeout: float) -> None:
    """Put the database in WAL, waiting out any connection doing the same.

    The journal mode belongs to the file, not to the connection, so only the
    first opener performs the transition. SQLite runs no busy handler for a
    journal_mode change, and while the winner holds the lock even *reading*
    the mode fails, so a single retry is not enough: wait and look again until
    the winner is done or the caller's timeout is spent.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            if str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal":
                return
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.005)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a write transaction. ``BEGIN IMMEDIATE`` takes the write lock up front."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def available_migrations() -> list[tuple[int, Path]]:
    migrations: list[tuple[int, Path]] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        match = _MIGRATION_PATTERN.match(path.name)
        if not match:
            raise ValueError(f"migration filename not in NNN_name.sql form: {path.name}")
        migrations.append((int(match.group(1)), path))
    versions = [version for version, _ in migrations]
    if len(set(versions)) != len(versions):
        raise ValueError(f"duplicate migration versions: {versions}")
    return migrations


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def latest_schema_version() -> int:
    migrations = available_migrations()
    return migrations[-1][0] if migrations else 0


@contextmanager
def _migration_lock(conn: sqlite3.Connection) -> Iterator[None]:
    """Serialize migrations across processes with a lock file.

    Two processes starting together both read user_version as 0 and both try to
    create the schema; the loser fails with "table dataset already exists".
    sqlite3.executescript commits any open transaction before it runs, so the
    version check cannot be made atomic inside SQLite itself — hence a file lock
    around the check and the apply.
    """
    row = conn.execute("PRAGMA database_list").fetchone()
    database = row[2] if row and row[2] else ""
    if not database or fcntl is None:
        yield
        return
    lock_path = Path(f"{database}.migrate.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply pending migrations in order. Returns the versions applied."""
    if schema_version(conn) >= latest_schema_version():
        return []
    with _migration_lock(conn):
        return _apply_pending(conn)


def _apply_pending(conn: sqlite3.Connection) -> list[int]:
    # Re-read under the lock: another process may have applied these already.
    current = schema_version(conn)
    applied: list[int] = []
    for version, path in available_migrations():
        if version <= current:
            continue
        script = path.read_text(encoding="utf-8")
        # executescript() commits any open transaction before it runs, so the
        # BEGIN/COMMIT have to live inside the script itself for the DDL and the
        # version bump to land together. user_version is transactional.
        conn.executescript(
            "BEGIN IMMEDIATE;\n"
            f"{script}\n"
            f"PRAGMA user_version = {int(version)};\n"
            "COMMIT;"
        )
        applied.append(version)
    return applied


def open_catalog(data_dir: Path) -> sqlite3.Connection:
    """Connect and bring the schema up to date. Safe to call from every process."""
    conn = connect(data_dir)
    migrate(conn)
    return conn


def catalog_for_process(data_dir: Path) -> sqlite3.Connection:
    """Thread-local connection, for callers that open one per request."""
    key = str(database_path(data_dir))
    cache: dict[str, sqlite3.Connection] = getattr(_local, "connections", None) or {}
    conn = cache.get(key)
    if conn is None:
        conn = open_catalog(data_dir)
        cache[key] = conn
        _local.connections = cache
    return conn


def close_process_connections() -> None:
    for conn in (getattr(_local, "connections", None) or {}).values():
        conn.close()
    _local.connections = {}


def read_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM index_meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def write_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO index_meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
