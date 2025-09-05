"""Run the Privatemode Document Store FastAPI app with Uvicorn.

Usage:
  python -m privatemode.document_store [--host 127.0.0.1] [--port 8081] [--reload] [--debug]

Environment variables:
  HOST (default: 127.0.0.1)
  PORT (default: 8081)
  RELOAD (default: false)
  LOG_LEVEL (default: info)
  CORS_ALLOW_ORIGINS (comma-separated list)
  VECTOR_DB_PATH (path to chroma persistence dir)
"""

from __future__ import annotations

import argparse
import os
import sys
import uvicorn


def str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    env_host = os.getenv("HOST", "127.0.0.1")
    env_port = int(os.getenv("PORT", "8081"))
    env_reload = str_to_bool(os.getenv("RELOAD"), default=False)

    parser = argparse.ArgumentParser(prog="python -m privatemode.document_store", add_help=True)
    parser.add_argument("--host", "-H", default=env_host, help="Bind host")
    parser.add_argument("--port", "-p", type=int, default=env_port, help="Bind port")
    parser.add_argument("--reload", "-r", action="store_true", default=env_reload, help="Enable autoreload")
    parser.add_argument("--log-level", "-l", default=os.getenv("LOG_LEVEL", "info"), help="Log level")
    parser.add_argument("--debug", action="store_true", help="Enable dev helpers (CORS allow localhost)")
    args = parser.parse_args(argv)

    if args.debug:
        os.environ["DEBUG_ALLOW_LOCALHOST"] = "1"

    uvicorn.run(
        "privatemode.document_store.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower(),
        factory=False,
    )


if __name__ == "__main__":
    main()
