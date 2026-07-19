from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

from ueba_server.config.app_config import create_app  # noqa: E402

app = create_app()


def main() -> None:
    os.chdir(SERVER_DIR)
    host = os.getenv("UEBA_HOST", "127.0.0.1")
    port = int(os.getenv("UEBA_PORT", "8000"))
    reload_enabled = os.getenv("UEBA_RELOAD", "0") == "1"
    uvicorn.run(
        "ueba_server.api:app",
        host=host,
        port=port,
        reload=reload_enabled,
        reload_dirs=[str(SERVER_DIR)] if reload_enabled else None,
    )


if __name__ == "__main__":
    main()
