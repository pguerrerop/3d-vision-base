from __future__ import annotations

import json
import time
from pathlib import Path

from vision_3d_acquisition.runtime.processing_queue import TakeProcessingQueue


def test_queue_append_claim_release_and_duplicate_prevention(tmp_path: Path) -> None:
    queue = TakeProcessingQueue(tmp_path / "data")
    first = queue.enqueue({"take_id": "take_1", "source_id": "s1", "modality": "rgb"})
    second = queue.enqueue({"take_id": "take_1", "source_id": "s1", "modality": "rgb"})
    assert first is not None
    assert second is None

    claim = queue.claim_next(worker_id="w1", modality="rgb")
    assert claim is not None
    assert claim["take_id"] == "take_1"
    queue.release_claim("take_1")
    status = queue.queue_status()
    assert status["active_claims_count"] == 0


def test_queue_stale_claim_recovery(tmp_path: Path) -> None:
    queue = TakeProcessingQueue(tmp_path / "data")
    queue.enqueue({"take_id": "take_2", "source_id": "s1", "modality": "heightmap"})
    claim = queue.claim_next(worker_id="w1", modality="heightmap", claim_timeout_sec=0.01)
    assert claim is not None
    time.sleep(0.02)
    stale = queue.recover_stale_claims(claim_timeout_sec=0.0)
    assert "take_2" in stale
    claim2 = queue.claim_next(worker_id="w2", modality="heightmap")
    assert claim2 is not None
    assert claim2["take_id"] == "take_2"


def test_queue_failure_retry_and_completion(tmp_path: Path) -> None:
    queue = TakeProcessingQueue(tmp_path / "data")
    queue.enqueue({"take_id": "take_3", "source_id": "s1", "modality": "rgb", "max_retries": 2})
    claim = queue.claim_next(worker_id="w1", modality="rgb")
    assert claim is not None
    retry = queue.mark_failed("take_3", {"error": "boom"})
    assert retry["retried"] is True
    claim_retry = queue.claim_next(worker_id="w1", modality="rgb")
    assert claim_retry is not None
    queue.mark_completed("take_3", {"status": "processing_completed"})
    status = queue.queue_status()
    assert status["completed_count"] == 1


def test_queue_persistence_files_exist(tmp_path: Path) -> None:
    queue = TakeProcessingQueue(tmp_path / "data")
    queue.enqueue({"take_id": "take_4", "source_id": "s1", "modality": "rgb"})
    queue.claim_next(worker_id="w1", modality="rgb")
    queue.mark_completed("take_4", {"status": "processing_completed"})
    assert queue.pending_path.is_file()
    assert queue.completed_path.is_file()
    assert queue.claims_path.is_file()
    claims = json.loads(queue.claims_path.read_text(encoding="utf-8"))
    assert claims == {}
