from __future__ import annotations

import json
import shutil
from pathlib import Path

from vision_3d_acquisition.contracts.metadata import AcquisitionMetadata
from vision_3d_acquisition.state.runtime_status import update_runtime_status
from vision_3d_acquisition.state.sessions import attach_take_to_session, ensure_session, generate_session_id
from vision_3d_acquisition.utils.paths import DataPaths

READY_MARKER = "READY"


class AcquisitionPublisher:
    """Publish takes to the filesystem incoming queue."""

    def __init__(self, data_dir: Path) -> None:
        self.paths = DataPaths(data_dir)
        self.paths.ensure_layout()

    def publish_take(
        self,
        take_id: str,
        metadata: AcquisitionMetadata,
        files: dict[str, Path],
    ) -> Path:
        """
        Stage, atomically publish, and signal a take.

        Args:
            take_id: Folder name under incoming/.
            metadata: Validated metadata (take_id must match).
            files: Map of FileReferences field names to source paths on disk.

        Returns:
            Path to the published take folder.
        """
        if metadata.take_id != take_id:
            raise ValueError(
                f"metadata.take_id {metadata.take_id!r} does not match take_id {take_id!r}"
            )

        session_id = metadata.session_id
        if session_id is None and metadata.mode == "live":
            session_id = generate_session_id()
            metadata = metadata.model_copy(update={"session_id": session_id})
        if session_id is not None:
            ensure_session(
                self.paths.root / "sessions",
                session_id,
                acquisition_mode=metadata.mode,
            )

        final_dir = self.paths.incoming_take(take_id)
        temp_dir = self.paths.incoming_temp(take_id)

        if final_dir.exists():
            raise FileExistsError(f"Take already exists: {final_dir}")
        if temp_dir.exists():
            raise FileExistsError(f"Staging folder already exists: {temp_dir}")

        temp_dir.mkdir(parents=True)
        try:
            self._copy_files(metadata, files, temp_dir)
            self._write_metadata(metadata, temp_dir)
            temp_dir.rename(final_dir)
            (final_dir / READY_MARKER).touch()
            self._update_acquisition_state(metadata)
            if session_id is not None:
                attach_take_to_session(
                    self.paths.root / "sessions",
                    session_id,
                    take_id,
                    take_metadata=metadata.model_dump(mode="json"),
                )
        except Exception:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            if final_dir.exists() and not (final_dir / READY_MARKER).exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            raise

        return final_dir

    def _copy_files(
        self,
        metadata: AcquisitionMetadata,
        files: dict[str, Path],
        dest_dir: Path,
    ) -> None:
        refs = metadata.files.model_dump()
        for key, filename in refs.items():
            if filename is None:
                continue
            if key not in files:
                raise ValueError(f"Missing source file for {key!r}")
            src = files[key]
            if not src.is_file():
                raise FileNotFoundError(f"Source file not found: {src}")
            shutil.copy2(src, dest_dir / filename)

    @staticmethod
    def _write_metadata(metadata: AcquisitionMetadata, dest_dir: Path) -> None:
        path = dest_dir / "metadata.json"
        path.write_text(
            json.dumps(metadata.model_dump(), indent=2) + "\n",
            encoding="utf-8",
        )

    def _update_acquisition_state(self, metadata: AcquisitionMetadata) -> None:
        source = _source_label(metadata.source)
        state = {
            "last_take_id": metadata.take_id,
            "last_published_at": metadata.created_at,
            "last_source": source,
            "last_mode": metadata.mode,
        }
        self.paths.acquisition_state.write_text(
            json.dumps(state, indent=2) + "\n",
            encoding="utf-8",
        )
        throughput = {}
        if isinstance(metadata.source, dict) and isinstance(metadata.source.get("fps"), (int, float)):
            throughput["acquisition_fps"] = metadata.source["fps"]
        update_runtime_status(
            self.paths.state,
            status="acquiring",
            latest_take_id=metadata.take_id,
            acquisition_connected=True,
            latest_frame_timestamp=metadata.created_at,
            current_session=metadata.session_id,
            acquisition_source=source,
            acquisition_source_details=metadata.source if isinstance(metadata.source, dict) else {"type": source},
            throughput=throughput,
            message="Acquisition frame received.",
        )


def _source_label(source: str | dict[str, object]) -> str:
    if isinstance(source, dict):
        return str(source.get("type") or "unknown")
    return source
