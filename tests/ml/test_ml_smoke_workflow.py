from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ml" / "run_ml_smoke_test.py"


def test_smoke_workflow_happy_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    cmd = [sys.executable, str(SCRIPT), "--data-dir", str(data_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "SMOKE TEST PASSED" in proc.stdout


def test_smoke_workflow_force_invalid_fallback(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    cmd = [sys.executable, str(SCRIPT), "--data-dir", str(data_dir), "--force-invalid-model"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "SMOKE TEST PASSED" in proc.stdout
    assert "fallback=True" in proc.stdout
