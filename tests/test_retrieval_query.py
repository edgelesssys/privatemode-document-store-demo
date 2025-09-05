from fastapi.testclient import TestClient

from privatemode.document_store.app import app
from privatemode.document_store.hybrid_db import tokenize, HybridDb
from tests.helper import setup_test

def test_retrieval_query_stores_then_retrieves(tmp_path):
    setup_test(tmp_path)

    # Use context manager to ensure app lifespan runs (DB initialized)
    with TestClient(app) as client:
        # 1) Store a document
        store_payload = {
            "id": "doc-kafka",
            "text": "Kafka uses backpressure by controlling fetch rates and consumer lag. " * 50,
            "metadata": {"title": "Operating Kafka at Scale", "namespace": "docs"},
            "collection": "test-retrieval",
        }
        r_store = client.post("/documents", json=store_payload, headers={"Authorization": "test-key"})
        assert r_store.status_code == 201

        # 2) Query for it
        query_payload = {
            "query": "How does backpressure work in Kafka?",
            "top_k": 5,
            "collection": "test-retrieval",
        }
        r = client.post("/retrieval/query", json=query_payload, headers={"Authorization": "test-key"})
        assert r.status_code == 200
        body = r.json()

        # Basic shape assertions
        assert set(body.keys()) >= {"took_ms", "hits", "exhaustive", "embedding_model", "index_version"}
        assert isinstance(body["hits"], list) and len(body["hits"]) >= 1

        # Verify at least one hit corresponds to the stored document
        doc_ids = {h.get("doc_id") for h in body["hits"]}
        assert "doc-kafka" in doc_ids

        # Check a hit structure
        hit = body["hits"][0]
        assert set(hit.keys()) >= {"doc_id", "chunk_id", "score", "raw_scores", "text", "offset", "metadata"}
        assert isinstance(hit["raw_scores"], dict)
        assert isinstance(hit["offset"], dict) and {"start", "end"} <= set(hit["offset"].keys())
        assert isinstance(hit["metadata"], dict)

        # Derived fields
        assert isinstance(body["embedding_model"], str) and len(body["embedding_model"]) > 0

        # 3) Delete the document
        r_delete = client.delete(f"/documents/test-retrieval/doc-kafka", headers={"Authorization": "test-key"})
        assert r_delete.status_code == 204

        # 4) Verify it's gone (404 on GET)
        r_get_after = client.get(f"/documents/test-retrieval/doc-kafka", headers={"Authorization": "test-key"})
        assert r_get_after.status_code == 404

        # 5) Query again should not find it
        r_query_after = client.post("/retrieval/query", json=query_payload, headers={"Authorization": "test-key"})
        assert r_query_after.status_code == 200
        body_after = r_query_after.json()
        doc_ids_after = {h.get("doc_id") for h in body_after["hits"]}
        assert "doc-kafka" not in doc_ids_after


def test_full_document_store_load_delete_without_embeddings(tmp_path):
    setup_test(tmp_path)
    # Use context manager to ensure app lifespan runs
    with TestClient(app) as client:
        collection = "test-full-docs"
        doc_id = "doc-no-embed"

        # 1) Store a full document without embeddings
        store_payload = {
            "id": doc_id,
            "text": "This is a full document stored without embeddings for ID-based retrieval only.",
            "metadata": {"title": "No Embed Doc", "namespace": "test"},
            "collection": collection,
            "embed": False
        }
        r_store = client.post("/documents", json=store_payload, headers={"Authorization": "test-key"})
        assert r_store.status_code == 201

        # 2) Retrieve the document
        r_get = client.get(f"/documents/{collection}/{doc_id}", headers={"Authorization": "test-key"})
        assert r_get.status_code == 200
        body = r_get.json()
        assert body["id"] == doc_id
        assert len(body["docs"]) == 1
        assert body["docs"][0] == store_payload["text"]
        assert len(body["metadata"]) == 1
        assert body["metadata"][0]['title'] == store_payload["metadata"]['title']
        assert body["metadata"][0]['namespace'] == store_payload["metadata"]['namespace']
        assert body["collection"] == collection

        # 3) Delete the document
        r_delete = client.delete(f"/documents/{collection}/{doc_id}", headers={"Authorization": "test-key"})
        assert r_delete.status_code == 204

        # 4) Verify it's gone (404 on GET)
        r_get_after = client.get(f"/documents/{collection}/{doc_id}", headers={"Authorization": "test-key"})
        assert r_get_after.status_code == 404

        # 5) Delete again should be idempotent (204 even if already deleted)
        r_delete = client.delete(f"/documents/{collection}/{doc_id}", headers={"Authorization": "test-key"})
        assert r_delete.status_code == 204


def test_split_document_chunking(tmp_path):
    """Test the split_document method for correct chunking with character limits and word overlap."""
    setup_test(tmp_path)

    # Create a vector DB and collection
    db = HybridDb(str(tmp_path / "data/test_chroma"), embedding_function_local=True)
    coll = db.create_collection("test-chunking", embed=True)

    # Test document that will exceed chunk size
    chunk_size = 128
    long_text = "word " * 500  # ~2500 chars, should split into chunks
    chunks = coll.split_document(long_text, chunk_size_tokens=chunk_size, overlap=5)

    # Verify chunks are created
    assert len(chunks) > 1, "Document should be split into multiple chunks"

    # Check chunk sizes are approximately correct
    for i, chunk in enumerate(chunks):
        assert len(tokenize(chunk)) <= chunk_size, f"Chunk {i} too large: {len(tokenize(chunk))} tokens"  # Allow some tolerance

    # Verify overlap: last words of chunk i should match first words of chunk i+1
    for i in range(len(chunks) - 1):
        prev_words = chunks[i].split()[-5:]  # Last 5 words of previous chunk
        next_words = chunks[i+1].split()[:5]  # First 5 words of next chunk
        assert prev_words == next_words, f"Overlap mismatch between chunk {i} and {i+1}"

    # Test edge case: short document (no splitting)
    short_text = "This is a short document."
    chunks_short = coll.split_document(short_text, chunk_size_tokens=chunk_size, overlap=5)
    assert len(chunks_short) == 1
    assert chunks_short[0] == short_text

    # Test with no overlap
    chunks_no_overlap = coll.split_document(long_text, chunk_size_tokens=chunk_size, overlap=0)
    assert len(chunks_no_overlap) > 1
    # No overlap check needed

    # Test long word slicing
    long_word = "a" * 2000  # 2000-char word
    chunks_long = coll.split_document(long_word, chunk_size_tokens=chunk_size, overlap=0)
    assert len(chunks_long) == 4

    # as we split a long word, the chunks may result in a slightly different number of tokens as we create new words
    assert len(tokenize(chunks_long[0])) <= chunk_size + 1
    assert len(tokenize(chunks_long[1])) <= chunk_size + 1
    assert len(tokenize(chunks_long[2])) <= chunk_size + 1
    assert len(tokenize(chunks_long[3])) == 18


def test_list_documents_with_limit(tmp_path):
    setup_test(tmp_path)
    with TestClient(app) as client:
        collection = "test-list-docs"

        # Store multiple documents
        docs = [
            {"id": "doc1", "text": "First document", "metadata": {"title": "Doc 1"}},
            {"id": "doc2", "text": "Second document", "metadata": {"title": "Doc 2"}},
            {"id": "doc3", "text": "Third document", "metadata": {"title": "Doc 3"}},
        ]

        for doc in docs:
            store_payload = {**doc, "collection": collection}
            r_store = client.post("/documents", json=store_payload, headers={"Authorization": "test-key"})
            assert r_store.status_code == 201

        # List all documents
        r_list_all = client.get(f"/documents/{collection}", headers={"Authorization": "test-key"})
        assert r_list_all.status_code == 200
        body_all = r_list_all.json()
        assert "documents" in body_all
        assert len(body_all["documents"]) == 3

        # List with limit=2
        r_list_limited = client.get(f"/documents/{collection}?limit=2", headers={"Authorization": "test-key"})
        assert r_list_limited.status_code == 200
        body_limited = r_list_limited.json()
        assert "documents" in body_limited
        assert len(body_limited["documents"]) == 2

        # Verify they are sorted by updated_at descending (most recent first)
        docs_all = body_all["documents"]
        docs_limited = body_limited["documents"]
        # The limited list should be the first 2 of the sorted all list
        assert docs_limited == docs_all[:2]
        # Specifically, the most recent should be doc3 and doc2
        assert [doc["id"] for doc in docs_limited] == ["doc3", "doc2"]

        assert docs_limited == docs_all[:2]
        # Specifically, the most recent should be doc3 and doc2
        assert [doc["id"] for doc in docs_limited] == ["doc3", "doc2"]

        # Check structure of returned documents
        for doc in docs_limited:
            assert "id" in doc
            assert "title" in doc
            assert "metadata" in doc
            assert "updated_at" in doc["metadata"]

        # Delete all documents
        for doc in docs:
            r_delete = client.delete(f"/documents/{collection}/{doc['id']}", headers={"Authorization": "test-key"})
            assert r_delete.status_code == 204

        # Verify the list is now empty
        r_list_after_delete = client.get(f"/documents/{collection}", headers={"Authorization": "test-key"})
        assert r_list_after_delete.status_code == 200
        body_after_delete = r_list_after_delete.json()
        assert body_after_delete["documents"] == []

def test_retrieval_query_advanced(tmp_path):
    setup_test(tmp_path)
    with TestClient(app) as client:
        # Store a document
        store_payload = {
            "id": "doc-kafka-advanced",
            "text": "Kafka uses backpressure by controlling fetch rates and consumer lag. " * 50,
            "metadata": {"title": "Operating Kafka at Scale", "namespace": "docs"},
            "collection": "test-retrieval-advanced",
        }
        r_store = client.post("/documents", json=store_payload, headers={"Authorization": "test-key"})
        assert r_store.status_code == 201

        # Query using advanced endpoint with messages
        messages = [
            {"role": "user", "content": "Tell me about databases"},
            {"role": "assistant", "content": "I can help with that"},
            {"role": "user", "content": "How does backpressure work in Kafka?"}
        ]
        advanced_query_payload = {
            "messages": messages,
            "context_doc": "Some context document",
            "top_k": 5,
            "collection": "test-retrieval-advanced",
        }
        r = client.post("/retrieval/query-advanced", json=advanced_query_payload, headers={"Authorization": "test-key"})
        assert r.status_code == 200
        body = r.json()

        # Basic shape assertions
        assert set(body.keys()) >= {"took_ms", "hits", "exhaustive", "embedding_model", "index_version"}
        assert isinstance(body["hits"], list) and len(body["hits"]) >= 1

        # Verify at least one hit corresponds to the stored document
        doc_ids = {h.get("doc_id") for h in body["hits"]}
        assert "doc-kafka-advanced" in doc_ids

        # Check a hit structure
        hit = body["hits"][0]
        assert set(hit.keys()) >= {"doc_id", "chunk_id", "score", "raw_scores", "text", "offset", "metadata"}
        assert isinstance(hit["raw_scores"], dict)
        assert isinstance(hit["offset"], dict) and {"start", "end"} <= set(hit["offset"].keys())
        assert isinstance(hit["metadata"], dict)

        # Derived fields
        assert isinstance(body["embedding_model"], str) and len(body["embedding_model"]) > 0

        # Clean up: Delete the document
        r_delete = client.delete(f"/documents/test-retrieval-advanced/doc-kafka-advanced", headers={"Authorization": "test-key"})
        assert r_delete.status_code == 204


def test_retrieval_query_invalid_api_key():
    # Use context manager to run startup/shutdown lifespan
    with TestClient(app) as client:
        payload = {
            "query": "test query",
            "top_k": 5,
            "collection": "test_invalid_key",
        }
        # first key is accepted & stored
        r = client.post("/retrieval/query", json=payload, headers={"Authorization": "init-key"})
        assert r.status_code == 404

        # second key is rejected
        r = client.post("/retrieval/query", json=payload, headers={"Authorization": "invalid-key"})
        assert r.status_code == 401
        body = r.json()
        assert body["code"] == "UNAUTHORIZED"
        assert "Invalid API Key" in body["message"]


def test_retrieval_query_no_api_key():
    # Use context manager to run startup/shutdown lifespan
    with TestClient(app) as client:
        payload = {
            "query": "test query",
            "top_k": 5,
            "collection": "test_no_key",
        }
        r = client.post("/retrieval/query", json=payload)
        assert r.status_code == 401
        body = r.json()
        assert body["code"] == "UNAUTHORIZED"
        assert "Invalid API Key" in body["message"]
