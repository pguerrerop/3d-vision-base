from __future__ import annotations

import argparse
import re
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_3d_acquisition.acquisition.trispector_25d import TriSpectorFtpAcquisitionAdapter
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.runtime.commanding import normalize_ftp_command, resolve_python_executable
from vision_3d_acquisition.runtime.logging import encode_event_line
from vision_3d_acquisition.runtime.models import RuntimeProcessEvent, utc_now_iso


@dataclass(slots=True)
class StabilityState:
    last_size: int = -1
    unchanged_checks: int = 0


class TriSpectorFtpRuntime:
    def __init__(
        self,
        *,
        process_id: str,
        source_id: str,
        ftp_command: str,
        upload_dir: Path,
        poll_interval_sec: float,
        stable_checks: int,
        auth_mode: str = "anonymous",
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.process_id = process_id
        self.source_id = source_id
        self.ftp_command = ftp_command
        self.upload_dir = upload_dir
        self.poll_interval_sec = max(0.1, float(poll_interval_sec))
        self.stable_checks = max(1, int(stable_checks))
        self.auth_mode = str(auth_mode or "anonymous").strip().lower()
        self.username = (username or "").strip() or None
        self.password = password or None
        self._settings = ApiSettings.from_env()
        self._adapter = TriSpectorFtpAcquisitionAdapter(self._settings, source_id=source_id)
        self._stop = False
        self._ftp_proc: subprocess.Popen[str] | None = None
        self._stability: dict[Path, StabilityState] = {}
        self._seen: set[Path] = set()
        self._last_heartbeat_monotonic = 0.0

        self.incoming_dir = self.upload_dir
        self.processing_dir = self.upload_dir.parent / "processing"
        self.processed_dir = self.upload_dir.parent / "processed"
        self.failed_dir = self.upload_dir.parent / "failed"
        for directory in (self.incoming_dir, self.processing_dir, self.processed_dir, self.failed_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def request_stop(self) -> None:
        self._stop = True

    def emit(self, event_type: str, message: str, *, severity: str = "info", metadata: dict[str, Any] | None = None) -> None:
        event = RuntimeProcessEvent(
            timestamp=utc_now_iso(),
            process_id=self.process_id,
            severity=severity,  # type: ignore[arg-type]
            event_type=event_type,
            message=message,
            metadata=metadata or {},
        )
        print(encode_event_line(event), flush=True)

    def run(self) -> int:
        self._start_ftp_server()
        self.emit("PROCESS_STARTED", "TriSpector FTP runtime started.", metadata={"source_id": self.source_id})
        try:
            while not self._stop:
                self._emit_heartbeat()
                self._check_ftp_health()
                self._scan_and_process_uploads()
                time.sleep(self.poll_interval_sec)
        finally:
            self._stop_ftp_server()
            self.emit("PROCESS_EXITED", "TriSpector FTP runtime stopped.", metadata={"source_id": self.source_id})
        return 0

    def _start_ftp_server(self) -> None:
        resolution = resolve_python_executable()
        if resolution.warning:
            self.emit("PYTHON_EXECUTABLE_FALLBACK", resolution.warning, severity="warning")
        normalized = normalize_ftp_command(
            self.ftp_command,
            self.upload_dir,
            python_executable=resolution.executable,
            auth_mode=self.auth_mode,
            username=self.username,
            password=self.password,
        )
        command = shlex.split(normalized.command.strip())
        if not command:
            raise ValueError("FTP command cannot be empty.")
        self._ftp_proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self._ftp_proc.stdout is not None:
            threading.Thread(target=self._ftp_stream_reader, args=("ftp_stdout", self._ftp_proc.stdout), daemon=True).start()
        if self._ftp_proc.stderr is not None:
            threading.Thread(target=self._ftp_stream_reader, args=("ftp_stderr", self._ftp_proc.stderr), daemon=True).start()
        if normalized.normalized_python:
            self.emit("FTP_COMMAND_NORMALIZED", "Normalized FTP command interpreter to resolved Python executable.")
        if normalized.injected_root:
            self.emit("FTP_ROOT_INJECTED", "Injected pyftpdlib upload root (-d) to match runtime upload directory.")
        if normalized.root_mismatch:
            self.emit(
                "FTP_ROOT_MISMATCH",
                "Configured pyftpdlib upload root differs from runtime upload directory.",
                severity="warning",
                metadata={"upload_dir": str(self.upload_dir), "command": normalized.command},
            )
        if normalized.injected_write:
            self.emit("FTP_WRITE_ENABLED", "Injected pyftpdlib write flag (-w).", metadata={"command": normalized.command})
        if normalized.auth_mode == "user_password":
            self.emit("FTP_AUTH_USER_PASSWORD", "FTP configured for user/password auth.", metadata={"username": self.username})
        else:
            self.emit("FTP_AUTH_ANONYMOUS", "FTP configured for anonymous auth.")
        self.emit(
            "FTP_SERVER_STARTED",
            "FTP server process started.",
            metadata={"pid": self._ftp_proc.pid, "command": normalized.command},
        )

    def _stop_ftp_server(self) -> None:
        if self._ftp_proc is None:
            return
        if self._ftp_proc.poll() is None:
            self._ftp_proc.terminate()
            try:
                self._ftp_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._ftp_proc.kill()
        self.emit(
            "FTP_SERVER_STOPPED",
            "FTP server process stopped.",
            metadata={"return_code": self._ftp_proc.returncode},
        )
        self._ftp_proc = None

    def _check_ftp_health(self) -> None:
        if self._ftp_proc is None:
            return
        return_code = self._ftp_proc.poll()
        if return_code is None:
            return
        self.emit(
            "FTP_SERVER_EXITED",
            "FTP server exited unexpectedly.",
            severity="error",
            metadata={"return_code": return_code},
        )
        self.request_stop()

    def _scan_and_process_uploads(self) -> None:
        uploads = sorted(path for path in self.incoming_dir.iterdir() if path.is_file())
        for upload in uploads:
            self._maybe_process_upload(upload)

    def _maybe_process_upload(self, upload: Path) -> None:
        if upload not in self._seen:
            self._seen.add(upload)
            self.emit("FILE_DETECTED", f"Detected upload {upload.name}.", metadata={"file": upload.name})

        size = upload.stat().st_size
        state = self._stability.setdefault(upload, StabilityState())
        if size > 0 and size == state.last_size:
            state.unchanged_checks += 1
        else:
            state.unchanged_checks = 0
            state.last_size = size

        if size <= 0 or state.unchanged_checks < self.stable_checks:
            return

        self.emit("FILE_STABLE", f"Upload {upload.name} is stable.", metadata={"file": upload.name, "size": size})
        self._stability.pop(upload, None)
        self._seen.discard(upload)
        self._process_stable_upload(upload)

    def _process_stable_upload(self, upload: Path) -> None:
        upload_size = upload.stat().st_size if upload.exists() else 0
        self.emit("FTP_UPLOAD_STARTED", f"Upload processing started for {upload.name}.", metadata={"filename": upload.name, "upload_size": upload_size})
        processing_path = self.processing_dir / upload.name
        upload.replace(processing_path)
        self.emit("FILE_MOVED_PROCESSING", f"Moved {upload.name} to processing.", metadata={"file": upload.name})
        try:
            result = self._adapter.parse_and_register_trispector_upload(processing_path)
        except Exception as exc:  # noqa: BLE001
            failed_path = self.failed_dir / processing_path.name
            processing_path.replace(failed_path)
            self.emit(
                "FTP_UPLOAD_FAILED",
                f"Failed to parse upload {processing_path.name}: {exc}",
                severity="error",
                metadata={"filename": processing_path.name, "upload_size": upload_size, "error": str(exc)},
            )
            return

        processed_path = self.processed_dir / processing_path.name
        processing_path.replace(processed_path)
        take_id = str(result.get("take_id") or "")
        processing_status = str(((result.get("processing") or {}) if isinstance(result.get("processing"), dict) else {}).get("status") or "")
        self.emit(
            "FTP_UPLOAD_COMPLETED",
            f"Upload completed for {processed_path.name}.",
            metadata={"filename": processed_path.name, "upload_size": upload_size, "take_id": take_id},
        )
        self.emit(
            "PARSE_SUCCESS",
            f"Parsed upload {processed_path.name}.",
            metadata={"file": processed_path.name, "take_id": take_id},
        )
        self.emit(
            "TAKE_REGISTERED",
            f"Registered take {take_id}.",
            metadata={"file": processed_path.name, "take_id": take_id, "source_id": self.source_id},
        )
        if processing_status:
            self.emit(
                "PROCESSING_TRIGGERED",
                f"Acquisition-processing integration status: {processing_status}",
                metadata={"take_id": take_id, "status": processing_status},
            )

    def _emit_heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat_monotonic < 5.0:
            return
        self._last_heartbeat_monotonic = now
        self.emit(
            "HEARTBEAT",
            "TriSpector FTP runtime heartbeat.",
            metadata={"upload_dir": str(self.incoming_dir), "poll_interval_sec": self.poll_interval_sec},
        )

    def _ftp_stream_reader(self, prefix: str, stream: Any) -> None:
        for line in iter(stream.readline, ""):
            text = line.rstrip("\n")
            print(f"[{prefix}] {text}", flush=True)
            self._emit_ftp_connection_event(text)
        stream.close()

    def _emit_ftp_connection_event(self, line: str) -> None:
        raw = line.strip()
        if not raw:
            return
        ip = _extract_ip(raw)
        username = _extract_username(raw)
        filename = _extract_filename(raw)
        size = _extract_size(raw)
        lower = raw.lower()
        if "connected" in lower:
            self.emit("FTP_CLIENT_CONNECTED", "FTP client connected.", metadata={"client_ip": ip})
        if "logged in" in lower:
            self.emit("FTP_CLIENT_LOGIN_SUCCESS", "FTP login success.", metadata={"client_ip": ip, "username": username})
        if "failed login" in lower or "login failed" in lower or "authentication failed" in lower:
            self.emit("FTP_CLIENT_LOGIN_FAILED", "FTP login failed.", severity="warning", metadata={"client_ip": ip, "username": username})
        if "stor" in lower and filename:
            self.emit("FTP_UPLOAD_STARTED", "FTP STOR command received.", metadata={"client_ip": ip, "username": username, "filename": filename, "upload_size": size})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime process worker entrypoint.")
    subparsers = parser.add_subparsers(dest="runtime_type", required=True)

    trispector = subparsers.add_parser("trispector_ftp_runtime", help="Run TriSpector FTP acquisition runtime.")
    trispector.add_argument("--process-id", default="trispector_ftp")
    trispector.add_argument("--source-id", default="trispector_ftp_0")
    trispector.add_argument("--ftp-command", required=True)
    trispector.add_argument("--upload-dir", required=True)
    trispector.add_argument("--poll-interval-sec", type=float, default=1.0)
    trispector.add_argument("--stable-checks", type=int, default=3)
    trispector.add_argument("--auth-mode", default="anonymous")
    trispector.add_argument("--username", default=None)
    trispector.add_argument("--password", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime: TriSpectorFtpRuntime | None = None
    if args.runtime_type == "trispector_ftp_runtime":
        runtime = TriSpectorFtpRuntime(
            process_id=str(args.process_id),
            source_id=str(args.source_id),
            ftp_command=str(args.ftp_command),
            upload_dir=Path(str(args.upload_dir)).expanduser().resolve(),
            poll_interval_sec=float(args.poll_interval_sec),
            stable_checks=int(args.stable_checks),
            auth_mode=str(args.auth_mode),
            username=(str(args.username) if args.username else None),
            password=(str(args.password) if args.password else None),
        )
    if runtime is None:
        raise ValueError(f"Unsupported runtime type: {args.runtime_type}")

    def _signal_handler(_signum: int, _frame: Any) -> None:
        runtime.request_stop()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    return runtime.run()


def _extract_ip(line: str) -> str | None:
    match = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", line)
    return match.group(1) if match else None


def _extract_username(line: str) -> str | None:
    match = re.search(r"user(?:name)?[ =:]+([\w\.-]+)", line, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _extract_filename(line: str) -> str | None:
    match = re.search(r"\bSTOR\s+([^\s]+)", line, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _extract_size(line: str) -> int | None:
    match = re.search(r"(\d+)\s*bytes", line, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
