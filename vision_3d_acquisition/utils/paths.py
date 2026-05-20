from pathlib import Path


class DataPaths:
    """Resolved paths under the data root."""

    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir.resolve()
        self.incoming = self.root / "incoming"
        self.processed = self.root / "processed"
        self.state = self.root / "state"
        self.acquisition_state = self.state / "acquisition.json"

    def incoming_take(self, take_id: str) -> Path:
        return self.incoming / take_id

    def incoming_temp(self, take_id: str) -> Path:
        return self.incoming / f".{take_id}.tmp"

    def ensure_layout(self) -> None:
        self.incoming.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)
