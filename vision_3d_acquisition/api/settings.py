from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from vision_3d_acquisition.api.env import env_port, load_env_file


DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_FRONTEND_PORT = 5173


@dataclass(frozen=True)
class ApiSettings:
    data_dir: Path
    incoming_dir: Path
    processed_dir: Path
    state_dir: Path
    events_dir: Path
    sessions_dir: Path
    datasets_dir: Path
    api_host: str = DEFAULT_API_HOST
    api_port: int = DEFAULT_API_PORT
    frontend_port: int = DEFAULT_FRONTEND_PORT
    cors_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "ApiSettings":
        load_env_file()
        data_dir = Path(os.environ.get("DATA_DIR", "data")).resolve()
        frontend_port = env_port("FRONTEND_PORT", DEFAULT_FRONTEND_PORT)
        settings = cls(
            data_dir=data_dir,
            incoming_dir=data_dir / "incoming",
            processed_dir=data_dir / "processed",
            state_dir=data_dir / "state",
            events_dir=data_dir / "events",
            sessions_dir=data_dir / "sessions",
            datasets_dir=data_dir / "datasets",
            api_host=os.environ.get("API_HOST", DEFAULT_API_HOST),
            api_port=env_port("API_PORT", DEFAULT_API_PORT),
            frontend_port=frontend_port,
            cors_origins=get_cors_origins(frontend_port),
        )
        settings.ensure_directories()
        return settings

    def ensure_directories(self) -> None:
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "runtime").mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings.from_env()


def get_cors_origins(frontend_port: int | None = None) -> tuple[str, ...]:
    load_env_file()
    configured = os.environ.get("CORS_ORIGINS")
    if configured:
        return tuple(origin.strip() for origin in configured.split(",") if origin.strip())

    port = frontend_port or env_port("FRONTEND_PORT", DEFAULT_FRONTEND_PORT)
    return (f"http://localhost:{port}", f"http://127.0.0.1:{port}")
