from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from vision_3d_acquisition.acquisition.process_integration import trigger_bound_processing_for_take
from vision_3d_acquisition.acquisition.replay_dataset import ReplayableAcquisitionService
from vision_3d_acquisition.api.settings import ApiSettings
from vision_3d_acquisition.contracts.metadata import AcquisitionMetadata, FileReferences, FrameSet
from vision_3d_acquisition.storage.publisher import AcquisitionPublisher
from vision_3d_acquisition.utils.ids import generate_take_id
from vision_3d_acquisition.utils.time import utc_now_iso


def parse_trispector_2_5d_image(input_file: Path, output_dir: Path) -> dict[str, Any]:
    arr = np.array(Image.open(input_file))
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale image, got shape {arr.shape}")
    h_full, w = arr.shape
    h = h_full // 3
    reflectance = arr[:h, :]
    payload = arr[h:, :]
    flat = payload.reshape(-1).astype(np.uint16)
    pairs = flat[: (flat.size // 2) * 2].reshape(-1, 2)
    height16 = (pairs[:, 1] * 256 + pairs[:, 0]).reshape(h, w)

    reflectance_path = output_dir / "reflectance.png"
    heightmap_path = output_dir / "height16.tif"
    preview_path = output_dir / "heightmap_preview.png"
    parser_metadata_path = output_dir / "parser_metadata.json"

    Image.fromarray(reflectance).save(reflectance_path)
    Image.fromarray(height16.astype(np.uint16)).save(heightmap_path)
    _write_heightmap_preview(height16, preview_path)
    valid_mask = np.isfinite(height16) & (height16 > 0)
    valid_vals = height16[valid_mask]
    raw_min = float(np.min(valid_vals)) if valid_vals.size else 0.0
    raw_max = float(np.max(valid_vals)) if valid_vals.size else 0.0
    p01 = float(np.percentile(valid_vals, 1)) if valid_vals.size else 0.0
    p99 = float(np.percentile(valid_vals, 99)) if valid_vals.size else 0.0
    unique_count = int(np.unique(valid_vals).size) if valid_vals.size else 0
    effective_range = max(0.0, raw_max - raw_min)
    effective_bits = float(np.log2(max(1.0, effective_range + 1.0)))
    encoding_warning: str | None = None
    if effective_bits <= 13.0:
        encoding_warning = "effective_range_low_for_uint16_possible_packed_12_14bit"
    parser_metadata = {
        "parser": "trispector_2_5d_image_v1",
        "input_shape": [int(h_full), int(w)],
        "output_shape": [int(h), int(w)],
        "has_reflectance": True,
        "height_encoding": "little_endian_uint16_pairs",
        "reconstruction_mode": "little_endian_uint16_default",
        "effective_bit_depth_stats": {
            "effective_bits_estimate": effective_bits,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "p01": p01,
            "p99": p99,
            "unique_value_count": unique_count,
            "warnings": [encoding_warning] if encoding_warning else [],
        },
        "height_preview_range_raw": {
            "min": raw_min,
            "max": raw_max,
            "valid_count": int(valid_vals.size),
            "units": "sensor_raw",
        },
    }
    parser_metadata_path.write_text(json.dumps(parser_metadata, indent=2), encoding="utf-8")
    return {
        "reflectance": reflectance_path,
        "heightmap": heightmap_path,
        "heightmap_preview": preview_path,
        "parser_metadata": parser_metadata_path,
        "diagnostics": parser_metadata,
    }


class TriSpectorFtpAcquisitionAdapter:
    def __init__(self, settings: ApiSettings, *, source_id: str = "trispector_ftp_0") -> None:
        self.settings = settings
        self.source_id = source_id
        self.publisher = AcquisitionPublisher(settings.data_dir)

    def parse_and_register_trispector_upload(self, upload_path: Path, *, take_id: str | None = None) -> dict[str, Any]:
        upload_path = upload_path.expanduser().resolve()
        if not upload_path.is_file():
            raise FileNotFoundError(f"TriSpector upload not found: {upload_path}")

        resolved_take_id = take_id or generate_take_id()
        created_at = utc_now_iso()
        with tempfile.TemporaryDirectory(prefix="trispector_25d_") as temp_dir:
            temp = Path(temp_dir)
            raw_name = f"raw_upload{upload_path.suffix.lower() or '.png'}"
            raw_upload_path = temp / raw_name
            raw_upload_path.write_bytes(upload_path.read_bytes())
            parsed = parse_trispector_2_5d_image(raw_upload_path, temp)
            metadata = AcquisitionMetadata(
                take_id=resolved_take_id,
                source={
                    "type": "ftp",
                    "sensor": "trispector_2_5d",
                    "source_id": self.source_id,
                    "acquisition_process_id": "trispector_ftp_acquisition",
                    "uploaded_filename": upload_path.name,
                    "height_preview_range_raw": (
                        (parsed.get("diagnostics") or {}).get("height_preview_range_raw")
                        if isinstance(parsed.get("diagnostics"), dict)
                        else None
                    ),
                },
                mode="live",
                created_at=created_at,
                frame_count=1,
                modalities=["heightmap", "reflectance"],  # type: ignore[arg-type]
                files=FileReferences(
                    heightmap="height16.tif",
                    reflectance="reflectance.png",
                    raw_upload=raw_name,
                    parser_metadata="parser_metadata.json",
                    heightmap_preview="heightmap_preview.png",
                ),
                frameset=FrameSet(
                    frameset_id=f"{resolved_take_id}_fs0",
                    timestamp=created_at,
                    frame_count=1,
                    synchronized=False,
                    timestamp_source="ftp_upload",
                ),
            )
            folder = self.publisher.publish_take(
                resolved_take_id,
                metadata,
                {
                    "heightmap": parsed["heightmap"],
                    "reflectance": parsed["reflectance"],
                    "raw_upload": raw_upload_path,
                    "parser_metadata": parsed["parser_metadata"],
                    "heightmap_preview": parsed["heightmap_preview"],
                },
            )
        replay_service = ReplayableAcquisitionService(self.settings.data_dir)
        source_payload = metadata.source if isinstance(metadata.source, dict) else {}
        session_ref = replay_service.resolve_session_for_take(source_metadata=source_payload)
        if session_ref is not None:
            replay_service.attach_take_to_session(
                take_id=resolved_take_id,
                dataset_id=session_ref.dataset_id,
                session_id=session_ref.session_id,
                source_metadata={
                    "session_id": session_ref.session_id,
                    "dataset_id": session_ref.dataset_id,
                    "created_at": created_at,
                },
            )
        replay_manifest = replay_service.write_replay_manifest(
            take_dir=folder,
            take_id=resolved_take_id,
            source_id=self.source_id,
            metadata=metadata.model_dump(mode="json"),
            parser_metadata=parsed.get("diagnostics") if isinstance(parsed.get("diagnostics"), dict) else {},
            session_ref=session_ref,
            acquisition_metadata={
                "uploaded_filename": upload_path.name,
                "upload_path": str(upload_path),
                "scan_direction": source_payload.get("scan_direction"),
                "profile_distance_mm": source_payload.get("profile_distance_mm"),
            },
        )
        processing = trigger_bound_processing_for_take(
            self.settings,
            take_id=resolved_take_id,
            source_id=self.source_id,
            modality="heightmap",
            purpose="acquisition_inspection",
        )
        return {
            "take_id": resolved_take_id,
            "folder": str(folder),
            "modality": "heightmap",
            "modality_label": "heightmap_2_5d",
            "replay_manifest": replay_manifest,
            "processing": processing,
        }


def _write_heightmap_preview(height16: np.ndarray, output_path: Path) -> None:
    values = np.asarray(height16, dtype=np.float32)
    finite = np.isfinite(values)
    positive = values > 0
    valid = finite & positive
    preview = np.zeros(values.shape, dtype=np.uint8)
    if np.any(valid):
        subset = values[valid]
        lo = float(np.min(subset))
        hi = float(np.max(subset))
        span = max(hi - lo, 1e-6)
        normalized = (values - lo) / span
        normalized = np.clip(normalized, 0.0, 1.0)
        preview = (normalized * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    cv2.imwrite(str(output_path), colored)
