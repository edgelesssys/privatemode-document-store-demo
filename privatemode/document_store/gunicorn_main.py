"""Gunicorn entrypoint for Privatemode Document Store.

Usage examples:

  gunicorn -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8081 \
    --workers 1 \
    privatemode.document_store.gunicorn_main:app

Environment variables honored by the underlying app module:
  HOST, PORT, RELOAD, LOG_LEVEL, CORS_ALLOW_ORIGINS, VECTOR_DB_PATH

Note: HOST/PORT here are typically provided via gunicorn arguments instead.
"""

from __future__ import annotations

import os
from typing import Any

from privatemode.document_store.app import app as app  # re-export for gunicorn

__all__ = ["app"]


def main() -> None:  # Optional: allow running module directly
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8081"))
    reload = os.getenv("RELOAD", "0").lower() in {"1", "true", "yes", "on"}
    uvicorn.run("privatemode.document_store.app:app", host=host, port=port, reload=reload, factory=False)


if __name__ == "__main__":  # pragma: no cover (manual invocation helper)
    main()
