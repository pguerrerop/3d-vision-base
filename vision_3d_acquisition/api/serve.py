from __future__ import annotations

import os

import uvicorn

from vision_3d_acquisition.api.env import env_bool, env_port, load_env_file
from vision_3d_acquisition.api.settings import DEFAULT_API_HOST, DEFAULT_API_PORT
from vision_3d_acquisition.runtime_config import warn_if_deprecated_processing_local_calibration


def main() -> None:
    load_env_file()
    warn_if_deprecated_processing_local_calibration()
    uvicorn.run(
        "vision_3d_acquisition.api.main:app",
        host=os.environ.get("API_HOST", DEFAULT_API_HOST),
        port=env_port("API_PORT", DEFAULT_API_PORT),
        reload=env_bool("API_RELOAD", False),
    )


if __name__ == "__main__":
    main()
