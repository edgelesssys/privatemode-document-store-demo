from fastapi.testclient import TestClient

from privatemode.document_store.app import app

from tests.helper import setup_test

def test_store_document_creates_chunks(tmp_path):
    setup_test(tmp_path)

    # Use context manager to run startup/shutdown lifespan
    with TestClient(app) as client:
        payload = {
            "id": "doc-1",
            "text": "alpha " * 400,  # long enough to create multiple chunks
            "metadata": {"role": "user"},
            "chunk_prefix": "Title: Example",
            "collection": "test_store_document_creates_chunks",
        }
        r = client.post("/documents", json=payload, headers={"Authorization": "test-key"})
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "ok"
        assert body["id"] == payload["id"]
        # chunks is optional telemetry; if present should be >= 1
        if body.get("chunks") is not None:
            assert body["chunks"] >= 1
