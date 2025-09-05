import json
import logging
import os
from typing import Optional

from fastapi import HTTPException, Request

logger = logging.getLogger("privatemode.document_store.auth")

_cached_api_key: Optional[str] = None
SMOKE_TEST_API_KEY = "test-key"  # for testing, allow this key without setting it

_allowed_origins = set()
_auth_key_path = None

def set_auth_key_path(path: str) -> None:
    global _auth_key_path, _cached_api_key
    _auth_key_path = path
    _cached_api_key = None

    load_api_key()

def set_allowed_origins(origins: list[str]) -> None:
    _allowed_origins.update(origins)

def load_api_key() -> Optional[str]:
    if _auth_key_path is None:
        return None
    api_key_file = os.path.join(_auth_key_path, "api_key.json")
    if not os.path.exists(api_key_file):
        logger.info("No API key loaded")
        return None

    try:
        with open(api_key_file, "r") as f:
            data = json.load(f)
            key = data.get("api_key")
            if key and isinstance(key, str) and key.strip():
                return key.strip()
    except (json.JSONDecodeError, IOError):
        logger.warning("Failed to load API key from %s", api_key_file)
    return None

def save_api_key(key: str) -> None:
    if _auth_key_path is None:
        return
    if not key or not key.strip():
        logger.warning("Attempted to save empty API key")
        return
    api_key_file = os.path.join(_auth_key_path, "api_key.json")
    try:
        with open(api_key_file, "w") as f:
            json.dump({"api_key": key.strip()}, f)
    except IOError:
        logger.warning("Failed to save API key to %s", api_key_file)

def verify_api_key(auth_header: str) -> None:
    # Verify API key, using cache
    global _cached_api_key
    if _cached_api_key is None:
        _cached_api_key = load_api_key()

    # if no api key is set, we store the first api key we get
    if _cached_api_key is None:
        # for testing, we allow a dummy key, which can only be used until a valid key is set
        if auth_header and auth_header.strip():
            stripped = auth_header.strip()
            if stripped == SMOKE_TEST_API_KEY:
                # allow dummy key for testing, until a key is set
                return

            _cached_api_key = stripped
            save_api_key(_cached_api_key)

    if _cached_api_key:
        if not auth_header or auth_header.strip() != _cached_api_key:
            raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED", "message": "Invalid API Key"})

def verify_origin(origin: str) -> None:
    if not origin:
        return
    if origin in _allowed_origins:
        return
    if origin.startswith("chrome-extension://"):
        return

    logger.warning("Blocked request from disallowed origin: %s", origin)
    raise HTTPException(status_code=403, detail={"code": "FORBIDDEN", "message": "Origin not allowed"})

def verify_auth(request: Request) -> None:
    verify_origin(request.headers.get("origin"))
    verify_api_key(request.headers.get("Authorization"))
