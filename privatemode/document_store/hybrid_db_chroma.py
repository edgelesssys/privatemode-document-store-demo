import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import hashlib
import logging
import os
import time
from typing import Any, cast, Dict, List, Optional
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
from chromadb.config import Settings
from chromadb.api.types import validate_embedding_function
from transformers import AutoTokenizer

from .fts import FTSStore, FTSChunk

logger = logging.getLogger("privatemode.document_store")
logger.info("Chroma version:", chromadb.__version__)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B')

def tokenize(text: str) -> List[int]:
    """Tokenize text using the multilingual E5 tokenizer."""
    return tokenizer.encode(text, add_special_tokens=False)

def detokenize(tokens: List[int]) -> str:
    """Detokenize tokens back to text."""
    return tokenizer.decode(tokens)

class Collection:
    def __init__(self, collection: Any, fts_store: FTSStore) -> None:
        # Using Any for chromadb collection to avoid tight coupling to internal module path
        self.collection: Any = collection
        self.fts_store = fts_store
        self.max_embedding_size = 1024  # the model support 32k; 1k to have smaller chunks

    def split_document(self, doc: str, chunk_size_tokens: int, overlap: int = 15) -> List[str]:
        """Split document into chunks using tokenization with overlap."""
        tokens = tokenize(doc)
        chunks = []
        chunk_sizes_min = float('inf')
        chunk_sizes_max = 0
        chunk_sizes_total = 0
        for i in range(0, len(tokens), chunk_size_tokens - overlap):
            chunk_tokens = tokens[i:i + chunk_size_tokens]
            chunk_sizes_min = min(chunk_sizes_min, len(chunk_tokens))
            chunk_sizes_max = max(chunk_sizes_max, len(chunk_tokens))
            chunk_sizes_total += len(chunk_tokens)
            chunk_text = detokenize(chunk_tokens)
            chunks.append(chunk_text)

        logger.info(f"Split document: {len(doc)/1024:.1f} kB, {len(chunks)} chunks, tokens: min={chunk_sizes_min}, max={chunk_sizes_max}, total={chunk_sizes_total}")
        return chunks

    def count(self) -> int:
        return self.collection.count()

    def upsert(self, id: str, text: str, metadata: Dict[str, Any], chunk_prefix: Optional[str] = None) -> int:
        start_time = time.time()
        doc_hash = hashlib.sha256(text.encode()).hexdigest()

        prefix_tokens = len(tokenize(f"{chunk_prefix}\n")) if chunk_prefix else 0
        chunk_size_tokens = self.max_embedding_size - prefix_tokens
        docs = self.split_document(text, chunk_size_tokens)

        split_end = time.time()
        if chunk_prefix:
            docs = [f"{chunk_prefix}\n{d}" for d in docs]

        # Note: Original code had 'docs = [docs[1]]' which seems like a bug; assuming it's meant to be all docs
        ids = [f"{id}-{i}" for i in range(len(docs))]
        metadatas = [metadata.copy() for _ in docs]
        for i, meta in enumerate(metadatas):
            meta["n"] = len(docs)
            meta["hash"] = doc_hash

        # only for embedded collections store the chunk for FTS
        if self.collection._embedding_function is not None:
            fts_chunks = []
            doc_id = f"{self.collection.name}-{id}"
            for i, (chunk_id, text) in enumerate(zip(ids, docs)):
                # Convert updated_at from ISO string to Unix timestamp for FTS
                fts_meta = metadatas[i].copy()
                if "updated_at" in fts_meta and isinstance(fts_meta["updated_at"], str):
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(fts_meta["updated_at"].replace('Z', '+00:00'))
                        fts_meta["updated_at"] = dt.timestamp()
                    except (ValueError, AttributeError):
                        pass  # Keep original if conversion fails

                fts_chunks.append(FTSChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    meta=fts_meta
                ))
            self.fts_store.upsert_chunks(fts_chunks)

        try:
            self.collection.upsert(documents=docs, metadatas=metadatas, ids=ids)
        except Exception as e:
            logger.error(f"Error upserting to collection '{self.collection.name}': {e}")
            raise

        end_time = time.time()
        logger.info(f"Upsert completed in {end_time - start_time:.2f} seconds, ({split_end - start_time:.2f}s for splitting)")
        return len(docs)

    def upsert_noembed(self, id: str, text: str, metadata: Dict[str, Any]) -> int:
        if self.collection._embedding_function is not None:
            raise ValueError("Use upsert() for collections with embeddings")
        doc_hash = hashlib.sha256(text.encode()).hexdigest()
        metadata["n"] = 1
        metadata["hash"] = doc_hash
        embeddings = [[0]]  # Dummy embedding to satisfy Chroma
        self.collection.upsert(documents=[text], embeddings=embeddings, metadatas=[metadata], ids=[id])
        return 1

    def get(self, id: Optional[str]) -> Dict:
        return self.collection.get(ids=id, include=["documents", "metadatas"])

    def delete(self, id: str) -> None:
        if self.collection._embedding_function is not None:
            self.fts_store.delete_doc(f"{self.collection.name}-{id}")

        if self.collection._embedding_function is None:
            self.collection.delete(ids=[id])
            return

        # if this is an embedded collection, delete all chunks
        meta = self.get_first(id)["metadatas"]
        if len(meta) == 0:
            return
        n = meta[0].get("n", 1)
        ids = [f"{id}-{i}" for i in range(n)]
        self.collection.delete(ids=ids)

    def get_hash(self, id: str) -> Optional[str]:
        meta = self.get_first(id)["metadatas"]
        if len(meta) == 0:
            return None
        return meta[0].get("hash", "")

    def get_first(self, id: str) -> Dict:
        return self.collection.get(ids=f"{id}-0", include=["metadatas"], limit=1)

    def exists(self, id: str, content: Optional[str] = None) -> bool:
        res = self.get_first(id)["metadatas"]
        if len(res) == 0:
            return False
        return (not content) or (hashlib.sha256(content.encode()).hexdigest() == res[0].get("hash", ""))

    def search(self, query: str, max_results: int = 10, where: Optional[Dict] = None, include: list = ["distances", "metadatas"]) -> Dict:
        docs = self.split_document(query, self.max_embedding_size)

        # ignore last chunk if too small
        if len(docs) > 1 and len(docs[-1].strip()) < 50:
            docs = docs[:-1]

        return self.collection.query(query_texts=docs, n_results=max_results, include=include, where=where)

    def search_perf(self, query: str, max_results: int = 10) -> tuple[Dict, float]:
        t0 = time.time()
        res = self.search(query, max_results)
        t1 = time.time()
        return res, (t1 - t0)

    def search_fts(self, query: str, limit: int = 20, with_snippet: bool = True, snippet_tokens: int = 16, updated_after: Optional[float] = None, updated_before: Optional[float] = None) -> List[Dict[str, Any]]:
        results = self.fts_store.search(query, limit, with_snippet, snippet_tokens, updated_after, updated_before, collection=self.collection.name)
        return results

# Combines Chroma and FTS to allow for both, vector search and keyword search.
class HybridDb:
    def __init__(self, path: str, embedding_function_local: bool = False) -> None:
        if embedding_function_local:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self.embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        else:
            model = "qwen3-embedding-4b"
            api_base = os.getenv("PRIVATEMODE_API_BASE", "http://localhost:8080/v1")
            api_key = os.getenv("PRIVATEMODE_API_KEY", "dummy")
            logger.info(f"Using OpenAI embedding function with model '{model}' and API base '{api_base}'")
            logger.info(f"API base used for embeddings: {api_base}")
            logger.info(f"API key length: {len(api_key)}")
            self.embedding_function = OpenAIEmbeddingFunction(api_key=api_key, model_name=model, api_base=api_base)
        validate_embedding_function(cast(Any, self.embedding_function))
        settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        )
        self.db = chromadb.PersistentClient(path=path, settings=settings)
        self.fts_store = FTSStore(os.path.join(path, "fts.db"))
        self.collections: Dict[str, Collection] = {}
        self.collections_metadata = self.db.create_collection(
            name="collections", embedding_function=None, get_or_create=True
        )

    def _get_embedding_function(self, coll_name) -> Any:
        docs = self.collections_metadata.get(ids=[coll_name], include=["metadatas"])
        if not docs:
            raise ValueError(f"Collection '{coll_name}' does not exist")
        metadatas = docs.get("metadatas", []) if docs else []
        has_embedding = bool(metadatas[0].get("has_embedding", False)) if metadatas else False
        return self.embedding_function if has_embedding else None

    def set_embedding_function(self, coll_name, embedding_fn: Any) -> None:
        has_embedding = embedding_fn is not None
        docs = self.collections_metadata.get(ids=[coll_name], include=["metadatas"])
        metadatas = docs.get("metadatas", []) if docs else []
        if not metadatas:
            meta = {"has_embedding": has_embedding}
            embeddings = [0]
            self.collections_metadata.upsert(ids=[coll_name], embeddings=embeddings, metadatas=[meta], documents=["<document>"])
            return

        current_has = metadatas[0].get("has_embedding", False)
        if current_has != has_embedding:
            raise ValueError(f"Collection '{coll_name}' already exists with different embedding setting: {current_has}")


    def get_collection(self, name: str, constructor: type = Collection) -> Collection:
        if name not in self.collections:
            try:
                # load collection from db; throws if no such collection
                embedding_func = self._get_embedding_function(name)
                db_collection = self.db.get_collection(name, embedding_function=embedding_func)
                self.collections[name] = constructor(db_collection, self.fts_store)
            except Exception:  # pragma: no cover - defensive
                raise ValueError(f"Collection '{name}' does not exist")
        return self.collections[name]

    def create_collection(self, name: str, embed: bool, constructor: type = Collection) -> Collection:
        embedding_function = cast(Any, self.embedding_function) if embed else None
        db_collection = self.db.create_collection(
            name=name, embedding_function=embedding_function, get_or_create=True
        )
        self.collections[name] = constructor(db_collection, self.fts_store)
        self.set_embedding_function(name, embedding_function)
        return self.collections[name]

    def reset(self) -> None:
        self.collections = {}
        self.db.reset()
