import pytest
import tempfile
import os
import time
from privatemode.document_store.fts import FTSStore, FTSChunk


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def store(temp_db):
    """Create an FTSStore instance."""
    store = FTSStore(temp_db)
    yield store
    store.close()


class TestFTSStore:
    def test_init_and_close(self, store):
        """Test initialization and closing."""
        assert store is not None
        store.close()
        # Should not raise an error

    def test_upsert_and_search(self, store):
        """Test upserting chunks and searching."""
        chunks = [
            FTSChunk(
                chunk_id="chunk1",
                doc_id="doc1",
                text="Python is a programming language",
                title="Python Intro",
                meta={"updated_at": time.time()}
            ),
            FTSChunk(
                chunk_id="chunk2",
                doc_id="doc1",
                text="JavaScript is also a programming language",
                title="JS Intro",
                meta={"updated_at": time.time()}
            ),
            FTSChunk(
                chunk_id="chunk3",
                doc_id="doc2",
                text="Machine learning with Python",
                title="ML Python",
                meta={"updated_at": time.time()}
            )
        ]

        # Upsert chunks
        count = store.upsert_chunks(chunks)
        assert count == 3

        # Search for "Python"
        results = store.search("Python")
        assert len(results) == 2
        assert results[0]["chunk_id"] in ["chunk1", "chunk3"]
        assert "Python" in results[0]["text"]

        # Search for "JavaScript"
        results = store.search("JavaScript")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "chunk2"

        # Search for non-existent term
        results = store.search("nonexistent")
        assert len(results) == 0

    def test_search_with_snippet(self, store):
        """Test search with snippets."""
        chunk = FTSChunk(
            chunk_id="chunk1",
            doc_id="doc1",
            text="This is a long text about Python programming and development",
            meta={"updated_at": time.time()}
        )
        store.upsert_chunks([chunk])

        results = store.search("Python", with_snippet=True)
        assert len(results) == 1
        assert results[0]["snippet"] is not None
        assert "<b>" in results[0]["snippet"]  # Highlighting tags

    def test_search_hyphenated_query(self, store):
        """Hyphenated keywords should not crash FTS5 query parsing."""
        chunk = FTSChunk(
            chunk_id="chunk1",
            doc_id="doc1",
            text="Qwen3 Embedding 4B is an embedding model",
            meta={"updated_at": time.time()},
        )
        store.upsert_chunks([chunk])

        results = store.search("Qwen3-Embedding-4B")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "chunk1"

    def test_search_limit(self, store):
        """Test search with limit."""
        chunks = [
            FTSChunk(
                chunk_id=f"chunk{i}",
                doc_id=f"doc{i}",
                text=f"Text with keyword {i}",
                meta={"updated_at": time.time()}
            ) for i in range(10)
        ]
        store.upsert_chunks(chunks)

        results = store.search("keyword", limit=5)
        assert len(results) == 5

    def test_time_based_filtering(self, store):
        """Test time-based filtering."""
        now = time.time()
        past = now - 3600  # 1 hour ago
        future = now + 3600  # 1 hour from now

        chunks = [
            FTSChunk(
                chunk_id="chunk1",
                doc_id="doc1",
                text="Recent content",
                meta={"updated_at": now}
            ),
            FTSChunk(
                chunk_id="chunk2",
                doc_id="doc2",
                text="Old content",
                meta={"updated_at": past}
            ),
            FTSChunk(
                chunk_id="chunk3",
                doc_id="doc3",
                text="Future content",
                meta={"updated_at": future}
            )
        ]
        store.upsert_chunks(chunks)

        # Search with updated_after
        results = store.search("content", updated_after=now - 1800)  # 30 min ago
        assert len(results) == 2  # Recent and future
        chunk_ids = {r["chunk_id"] for r in results}
        assert "chunk1" in chunk_ids
        assert "chunk3" in chunk_ids

        # Search with updated_before
        results = store.search("content", updated_before=now + 1800)  # 30 min from now
        assert len(results) == 2  # Old and recent
        chunk_ids = {r["chunk_id"] for r in results}
        assert "chunk1" in chunk_ids
        assert "chunk2" in chunk_ids

        # Search within range
        results = store.search("content", updated_after=past + 1800, updated_before=future - 1800)
        assert len(results) == 1  # Only recent
        assert results[0]["chunk_id"] == "chunk1"

    def test_delete_doc(self, store):
        """Test deleting a document."""
        chunks = [
            FTSChunk(
                chunk_id="chunk1",
                doc_id="doc1",
                text="Content of doc1",
                meta={"updated_at": time.time()}
            ),
            FTSChunk(
                chunk_id="chunk2",
                doc_id="doc1",
                text="More content of doc1",
                meta={"updated_at": time.time()}
            ),
            FTSChunk(
                chunk_id="chunk3",
                doc_id="doc2",
                text="Content of doc2",
                meta={"updated_at": time.time()}
            )
        ]
        store.upsert_chunks(chunks)

        # Verify all chunks are searchable
        results = store.search("content")
        assert len(results) == 3

        # Delete doc1
        deleted = store.delete_doc("doc1")
        assert deleted == 2

        # Only doc2 should remain
        results = store.search("content")
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc2"

    def test_delete_chunks(self, store):
        """Test deleting specific chunks."""
        chunks = [
            FTSChunk(
                chunk_id="chunk1",
                doc_id="doc1",
                text="Content 1",
                meta={"updated_at": time.time()}
            ),
            FTSChunk(
                chunk_id="chunk2",
                doc_id="doc1",
                text="Content 2",
                meta={"updated_at": time.time()}
            ),
            FTSChunk(
                chunk_id="chunk3",
                doc_id="doc2",
                text="Content 3",
                meta={"updated_at": time.time()}
            )
        ]
        store.upsert_chunks(chunks)

        # Delete specific chunks
        deleted = store.delete_chunks(["chunk1", "chunk3"])
        assert deleted == 2

        # Only chunk2 should remain
        results = store.search("Content")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "chunk2"

    def test_empty_search(self, store):
        """Test searching with empty query."""
        results = store.search("")
        assert results == []

        results = store.search("   ")
        assert results == []

    def test_metadata_storage(self, store):
        """Test that metadata is stored and retrieved correctly."""
        meta = {
            "updated_at": time.time(),
            "author": "Test Author",
            "tags": ["test", "metadata"]
        }
        chunk = FTSChunk(
            chunk_id="chunk1",
            doc_id="doc1",
            text="Test content",
            title="Test Title",
            tags=["tag1", "tag2"],
            meta=meta
        )
        store.upsert_chunks([chunk])

        results = store.search("content")
        assert len(results) == 1
        result = results[0]
        assert result["title"] == "Test Title"
        assert result["tags"] == "tag1,tag2"
        assert result["meta"] == meta
