import glob
import os
import uuid

import pytest

from privatemode.document_store.hybrid_db import Collection, HybridDb


def _populate(collection: Collection):
    doc1 = (
        "Elasticsearch with a Vector Plugin: Elasticsearch is a highly scalable search engine "
        "that supports full-text search. By default, it does not support vector search, but there "
        "are plugins available (such as the Elasticsearch vector scoring plugin) that add this capability. "
        "Elasticsearch is well-suited for environments where data is continuously updated."
    )
    doc2 = (
        "For your use case, where you need a database that supports vector search for text snippets and allows "
        "continuous updating with new data, you would typically look for a 'vector database' specialized in "
        "handling vector embeddings for similarity search."
    )
    doc3 = (
        "Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA, is a member of the "
        "programming and chess clubs who enjoys pizza, swimming, and hiking in her free time in hopes of working "
        "at a tech company after graduating from the University of Washington."
    )
    doc4 = (
        "The university chess club provides an outlet for students to come together and enjoy playing the classic "
        "strategy game of chess. Members of all skill levels are welcome, from beginners learning the rules to "
        "experienced tournament players."
    )
    doc5 = (
        "The University of Washington, founded in 1861 in Seattle, is a public research university with over 45,000 "
        "students across three campuses in Seattle, Tacoma, and Bothell."
    )

    collection.upsert("1", doc1, {"role": "user"})
    collection.upsert("2", doc2, {"role": "assistant"})
    collection.upsert("3", doc3, {"role": "assistant"})
    collection.upsert("4", doc4, {"role": "assistant"})
    collection.upsert("5", doc5, {"role": "assistant"})


def test_vector_db_add_and_search(tmp_path):
    db_path = tmp_path / "chroma"
    db = HybridDb(str(db_path), embedding_function_local=True)
    coll_name = f"test_{uuid.uuid4().hex[:8]}"
    coll = db.create_collection(coll_name, embed=True)

    assert coll.count() == 0
    _populate(coll)
    assert coll.count() > 0

    res = coll.search(
        "I want to use a vector db to find text snippets similar to some search text. "
        "The content of the db must be continuously updated with new data. What db shall I use?",
        max_results=5,
    )
    # Expect chroma-like shape with lists-per-query
    assert isinstance(res, dict)
    # Ensure at least metadatas present and non-empty
    assert "metadatas" in res and len(res["metadatas"]) >= 1
    assert isinstance(res["metadatas"][0], list) and len(res["metadatas"][0]) >= 1

    db.reset()

    # expect an exception when fetching a deleted collection
    with pytest.raises(ValueError):
        db.get_collection(coll_name)


def test_vector_db_no_embed(tmp_path):
    db_path = tmp_path / "chroma"
    db = HybridDb(str(db_path), embedding_function_local=True)
    coll_name = f"test_noembed_{uuid.uuid4().hex[:8]}"
    coll = db.create_collection(coll_name, embed=False)

    assert coll.count() == 0

    # Add a full document without chunking or embeddings
    doc_id = "fulldoc1"
    full_text = "This is a full document about the University of Washington and chess."
    metadata = {"role": "user", "source": "test"}
    coll.upsert_noembed(doc_id, full_text, metadata)
    assert coll.count() == 1

    # Retrieve the full document by ID
    retrieved = coll.get(doc_id)
    assert retrieved is not None
    assert len(retrieved["documents"]) == 1
    assert retrieved["documents"][0] == full_text
    assert retrieved["metadatas"][0]["role"] == "user"
    assert retrieved["metadatas"][0]["source"] == "test"

    db.reset()

    # expect an exception when fetching a deleted collection
    with pytest.raises(ValueError):
        db.get_collection(coll_name)
