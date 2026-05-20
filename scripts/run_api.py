#!/usr/bin/env python3
"""Start the FastAPI backend (loads .env: API_HOST, API_PORT, API_RELOAD)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision_3d_acquisition.api.serve import main  # noqa: E402

if __name__ == "__main__":
    main()
