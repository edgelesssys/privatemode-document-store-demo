import json
import os
import pytest
from unittest.mock import Mock

from fastapi import HTTPException
from privatemode.document_store.auth import (
    set_auth_key_path,
    set_allowed_origins,
    load_api_key,
    save_api_key,
    verify_api_key,
    verify_origin,
    verify_auth,
    SMOKE_TEST_API_KEY,
)


class MockRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}


@pytest.fixture
def temp_auth_key_path(tmp_path):
    auth_key_path = tmp_path / "test_db"
    auth_key_path.mkdir()
    set_auth_key_path(str(auth_key_path))
    return auth_key_path


def test_set_auth_key_path(temp_auth_key_path):
    # Already set in fixture
    assert load_api_key() is None  # No file yet


def test_load_api_key_no_file(temp_auth_key_path):
    assert load_api_key() is None


def test_load_api_key_with_file(temp_auth_key_path):
    api_key_file = temp_auth_key_path / "api_key.json"
    with open(api_key_file, "w") as f:
        json.dump({"api_key": "test-key"}, f)
    assert load_api_key() == "test-key"


def test_load_api_key_invalid_json(temp_auth_key_path):
    api_key_file = temp_auth_key_path / "api_key.json"
    with open(api_key_file, "w") as f:
        f.write("invalid json")
    assert load_api_key() is None


def test_save_api_key(temp_auth_key_path):
    save_api_key("new-key")
    api_key_file = temp_auth_key_path / "api_key.json"
    assert api_key_file.exists()
    with open(api_key_file, "r") as f:
        data = json.load(f)
        assert data["api_key"] == "new-key"


def test_save_api_key_empty(temp_auth_key_path):
    save_api_key("")
    api_key_file = temp_auth_key_path / "api_key.json"
    assert not api_key_file.exists()


def test_verify_api_key_no_key(temp_auth_key_path):
    verify_api_key(str(temp_auth_key_path))  # Should not raise


def test_verify_api_key_smoke_test_key(temp_auth_key_path):
    verify_api_key(SMOKE_TEST_API_KEY)  # Should not raise


def test_verify_api_key_set_new_key(temp_auth_key_path):
    verify_api_key("new-key")  # Should set and save
    assert load_api_key() == "new-key"


def test_verify_api_key_cached(temp_auth_key_path):
    # Set key first
    save_api_key("cached-key")
    verify_api_key("cached-key")  # Should not raise


def test_verify_api_key_invalid(temp_auth_key_path):
    save_api_key("valid-key")
    with pytest.raises(HTTPException) as exc:
        verify_api_key("invalid-key")
    assert exc.value.status_code == 401


def test_verify_origin_allowed():
    set_allowed_origins(["http://example.com"])
    verify_origin("http://example.com")  # Should not raise


def test_verify_origin_blocked():
    with pytest.raises(HTTPException) as exc:
        verify_origin("http://blocked.com")
    assert exc.value.status_code == 403


def test_verify_origin_chrome_extension():
    verify_origin("chrome-extension://abc123")  # Should allow all chrome


def test_verify_auth_success(temp_auth_key_path):
    set_allowed_origins(["http://example.com"])
    save_api_key("test-key")
    request = MockRequest({"origin": "http://example.com", "Authorization": "test-key"})
    verify_auth(request)  # Should not raise


def test_verify_auth_origin_blocked(temp_auth_key_path):
    request = MockRequest({"origin": "http://blocked.com"})
    with pytest.raises(HTTPException) as exc:
        verify_auth(request)
    assert exc.value.status_code == 403


def test_verify_auth_api_key_invalid(temp_auth_key_path):
    set_allowed_origins(["http://example.com"])
    save_api_key("valid-key")
    request = MockRequest({"origin": "http://example.com", "Authorization": "invalid-key"})
    with pytest.raises(HTTPException) as exc:
        verify_auth(request)
    assert exc.value.status_code == 401
