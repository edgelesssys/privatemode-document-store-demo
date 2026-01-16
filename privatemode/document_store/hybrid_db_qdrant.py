import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import tiktoken
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVector, SparseVectorParams, VectorParams

from .fts import FTSChunk, FTSStore

logger = logging.getLogger("privatemode.document_store")

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _tokenize(text: str) -> List[int]:
    return _TOKENIZER.encode(text or "")


def _detokenize(tokens: Sequence[int]) -> str:
    return _TOKENIZER.decode(list(tokens))


def _stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


@dataclass(frozen=True)
class SparseEncoding:
    indices: List[int]
    values: List[float]


class SparseEncoder:
    def __init__(self, dim: int = 65536) -> None:
        self.dim = dim

    def encode(self, text: str) -> SparseEncoding:
        tokens = re.findall(r"[a-zA-Z0-9]+", _normalize_text(text))
        if not tokens:
            return SparseEncoding(indices=[], values=[])

        counts: Dict[int, int] = {}
        for token in tokens:
            idx = _stable_hash(token) % self.dim
            counts[idx] = counts.get(idx, 0) + 1

        indices = sorted(counts.keys())
        values = [1.0 + float(math.log1p(counts[i])) for i in indices]
        return SparseEncoding(indices=indices, values=values)


class DenseEmbedder:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def dimension(self) -> int:
        raise NotImplementedError


class SentenceTransformerEmbedder(DenseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def dimension(self) -> int:
        return self._dim


class OpenAIEmbedder(DenseEmbedder):
    def __init__(self, model: str, api_base: str, api_key: str) -> None:
        import openai

        self._model = model
        self._client = openai.OpenAI(api_key=api_key, base_url=api_base)
        self._dim: Optional[int] = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        vectors = [d.embedding for d in response.data]
        if self._dim is None and vectors:
            self._dim = len(vectors[0])
        return vectors

    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_documents(["dimension probe"])[0])
        return int(self._dim)


class CollectionMetaStore:
    def __init__(self, base_path: str) -> None:
        self._path = os.path.join(base_path, "collections.json")
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
            except Exception:
                self._data = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh)

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._data.get(name)

    def set(self, name: str, meta: Dict[str, Any]) -> None:
        self._data[name] = meta
        self._save()

    def delete(self, name: str) -> None:
        if name in self._data:
            del self._data[name]
            self._save()

    def list_names(self) -> List[str]:
        return list(self._data.keys())


class Collection:
    def __init__(
        self,
        name: str,
        client: QdrantClient,
        fts_store: FTSStore,
        embedder: Optional[DenseEmbedder],
        sparse_encoder: Optional[SparseEncoder],
        embed_enabled: bool,
    ) -> None:
        self.name = name
        self.client = client
        self.fts_store = fts_store
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.embed_enabled = embed_enabled
        self.max_embedding_size = 1024

    def split_document(self, doc: str, chunk_size_tokens: int, overlap: int = 15) -> List[str]:
        tokens = _tokenize(doc)
        chunks: List[str] = []
        chunk_sizes_min = float("inf")
        chunk_sizes_max = 0
        chunk_sizes_total = 0
        if chunk_size_tokens <= 0:
            return [doc]
        step = max(1, chunk_size_tokens - overlap)
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + chunk_size_tokens]
            chunk_sizes_min = min(chunk_sizes_min, len(chunk_tokens))
            chunk_sizes_max = max(chunk_sizes_max, len(chunk_tokens))
            chunk_sizes_total += len(chunk_tokens)
            chunk_text = _detokenize(chunk_tokens)
            chunks.append(chunk_text)

        if chunks:
            logger.info(
                "Split document: %.1f kB, %s chunks, tokens: min=%s, max=%s, total=%s",
                len(doc) / 1024,
                len(chunks),
                chunk_sizes_min,
                chunk_sizes_max,
                chunk_sizes_total,
            )
        return chunks

    def count(self) -> int:
        res = self.client.count(collection_name=self.name, exact=True)
        return int(res.count or 0)

    def _payload_from_metadata(self, metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
        payload = dict(metadata or {})
        payload["text"] = text
        return payload

    def _split_and_prepare(self, text: str, metadata: Dict[str, Any], chunk_prefix: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        prefix_tokens = len(_tokenize(f"{chunk_prefix}\n")) if chunk_prefix else 0
        chunk_size_tokens = self.max_embedding_size - prefix_tokens
        docs = self.split_document(text, chunk_size_tokens)
        if chunk_prefix:
            docs = [f"{chunk_prefix}\n{d}" for d in docs]
        metadatas = [metadata.copy() for _ in docs]
        return docs, metadatas

    def upsert(self, id: str, text: str, metadata: Dict[str, Any], chunk_prefix: Optional[str] = None) -> int:
        if not self.embed_enabled:
            raise ValueError("Use upsert_noembed() for collections without embeddings")

        start_time = time.time()
        doc_hash = hashlib.sha256(text.encode()).hexdigest()
        docs, metadatas = self._split_and_prepare(text, metadata, chunk_prefix)

        for meta in metadatas:
            meta["n"] = len(docs)
            meta["hash"] = doc_hash

        if self.embedder is None:
            raise RuntimeError("Embedding function is not configured")

        dense_vectors = self.embedder.embed_documents(docs)
        sparse_vectors = [self.sparse_encoder.encode(d) if self.sparse_encoder else SparseEncoding([], []) for d in docs]

        points: List[models.PointStruct] = []
        for i, doc in enumerate(docs):
            chunk_id = f"{id}-{i}"
            payload = self._payload_from_metadata(metadatas[i], doc)
            points.append(
                models.PointStruct(
                    id=chunk_id,
                    vector={"dense": dense_vectors[i]},
                    sparse_vector=SparseVector(
                        indices=sparse_vectors[i].indices,
                        values=sparse_vectors[i].values,
                    ),
                    payload=payload,
                )
            )

        if points:
            self.client.upsert(collection_name=self.name, points=points)

        if self.embed_enabled:
            fts_chunks = []
            doc_id = f"{self.name}-{id}"
            for i, (chunk_id, doc) in enumerate(zip([p.id for p in points], docs)):
                fts_meta = metadatas[i].copy()
                if "updated_at" in fts_meta and isinstance(fts_meta["updated_at"], str):
                    try:
                        from datetime import datetime

                        dt = datetime.fromisoformat(fts_meta["updated_at"].replace("Z", "+00:00"))
                        fts_meta["updated_at"] = dt.timestamp()
                    except (ValueError, AttributeError):
                        pass
                fts_chunks.append(
                    FTSChunk(
                        chunk_id=str(chunk_id),
                        doc_id=doc_id,
                        text=doc,
                        meta=fts_meta,
                    )
                )
            if fts_chunks:
                self.fts_store.upsert_chunks(fts_chunks)

        end_time = time.time()
        logger.info(
            "Upsert completed in %.2f seconds",
            end_time - start_time,
        )
        return len(docs)

    def upsert_noembed(self, id: str, text: str, metadata: Dict[str, Any]) -> int:
        if self.embed_enabled:
            raise ValueError("Use upsert() for collections with embeddings")
        doc_hash = hashlib.sha256(text.encode()).hexdigest()
        metadata = dict(metadata or {})
        metadata["n"] = 1
        metadata["hash"] = doc_hash
        payload = self._payload_from_metadata(metadata, text)
        point = models.PointStruct(id=id, vector={"dense": [0.0]}, payload=payload)
        self.client.upsert(collection_name=self.name, points=[point])
        return 1

    def _convert_where(self, where: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        if not where:
            return None
        must = []
        for key, value in where.items():
            if isinstance(value, dict):
                range_cond = models.Range(
                    gte=value.get("$gte"),
                    lte=value.get("$lte"),
                    gt=value.get("$gt"),
                    lt=value.get("$lt"),
                )
                must.append(models.FieldCondition(key=key, range=range_cond))
            else:
                must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        return models.Filter(must=must)

    def _points_to_chroma(self, points: Sequence[models.ScoredPoint]) -> Dict[str, List[List[Any]]]:
        ids: List[str] = []
        distances: List[float] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []

        for p in points:
            payload = dict(p.payload or {})
            text = payload.pop("text", "")
            ids.append(str(p.id))
            documents.append(text)
            metadatas.append(payload)
            score = float(p.score or 0.0)
            distance = 1.0 / (1.0 + score)
            distances.append(distance)

        return {
            "ids": [ids],
            "distances": [distances],
            "metadatas": [metadatas],
            "documents": [documents],
        }

    def search(
        self,
        query: str,
        max_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: List[str] = ["distances", "metadatas"],
    ) -> Dict[str, Any]:
        docs = self.split_document(query, self.max_embedding_size)
        if len(docs) > 1 and len(docs[-1].strip()) < 50:
            docs = docs[:-1]

        if not docs:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

        if not self.embed_enabled or self.embedder is None:
            raise ValueError("Search requires embeddings")

        filter_obj = self._convert_where(where)
        results: List[Dict[str, List[List[Any]]]] = []
        for doc in docs:
            dense_query = self.embedder.embed_query(doc)
            sparse_query = self.sparse_encoder.encode(doc) if self.sparse_encoder else SparseEncoding([], [])

            try:
                result = self.client.query_points(
                    collection_name=self.name,
                    query=models.FusionQuery(
                        prefetch=[
                            models.Prefetch(
                                query=dense_query,
                                using="dense",
                                limit=max_results * 2,
                                filter=filter_obj,
                            ),
                            models.Prefetch(
                                query=SparseVector(indices=sparse_query.indices, values=sparse_query.values),
                                using="sparse",
                                limit=max_results * 2,
                                filter=filter_obj,
                            ),
                        ],
                        fusion=models.Fusion.RRF,
                    ),
                    limit=max_results,
                    with_payload=True,
                )
                points = cast(Sequence[models.ScoredPoint], getattr(result, "points", result))
            except Exception:
                points = self.client.search(
                    collection_name=self.name,
                    query_vector=models.NamedVector(name="dense", vector=dense_query),
                    limit=max_results,
                    with_payload=True,
                    query_filter=filter_obj,
                )

            results.append(self._points_to_chroma(points))

        merged = {
            "ids": [],
            "distances": [],
            "metadatas": [],
            "documents": [],
        }
        for res in results:
            merged["ids"].append(res["ids"][0])
            merged["distances"].append(res["distances"][0])
            merged["metadatas"].append(res["metadatas"][0])
            merged["documents"].append(res["documents"][0])
        return merged

    def search_perf(self, query: str, max_results: int = 10) -> tuple[Dict[str, Any], float]:
        t0 = time.time()
        res = self.search(query, max_results)
        t1 = time.time()
        return res, (t1 - t0)

    def search_fts(
        self,
        query: str,
        limit: int = 20,
        with_snippet: bool = True,
        snippet_tokens: int = 16,
        updated_after: Optional[float] = None,
        updated_before: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return self.fts_store.search(
            query,
            limit,
            with_snippet,
            snippet_tokens,
            updated_after,
            updated_before,
            collection=self.name,
        )

    def get(self, id: Optional[str]) -> Dict[str, Any]:
        if id is None:
            points: List[models.Record] = []
            offset = None
            while True:
                res = self.client.scroll(
                    collection_name=self.name,
                    offset=offset,
                    limit=100,
                    with_payload=True,
                )
                points.extend(res[0])
                offset = res[1]
                if offset is None:
                    break
        else:
            points = self.client.retrieve(collection_name=self.name, ids=[id], with_payload=True)

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        for p in points:
            payload = dict(p.payload or {})
            text = payload.pop("text", "")
            ids.append(str(p.id))
            docs.append(text)
            metas.append(payload)

        return {"ids": ids, "documents": docs, "metadatas": metas}

    def get_first(self, id: str) -> Dict[str, Any]:
        return self.get(f"{id}-0")

    def get_hash(self, id: str) -> Optional[str]:
        meta = self.get_first(id).get("metadatas", [])
        if not meta:
            return None
        return cast(Optional[str], meta[0].get("hash"))

    def exists(self, id: str, content: Optional[str] = None) -> bool:
        meta = self.get_first(id).get("metadatas", [])
        if not meta:
            return False
        if not content:
            return True
        return hashlib.sha256(content.encode()).hexdigest() == meta[0].get("hash")

    def delete(self, id: str) -> None:
        if self.embed_enabled:
            self.fts_store.delete_doc(f"{self.name}-{id}")
            meta = self.get_first(id).get("metadatas", [])
            if not meta:
                return
            n = int(meta[0].get("n", 1))
            ids = [f"{id}-{i}" for i in range(n)]
        else:
            ids = [id]

        self.client.delete(collection_name=self.name, points_selector=models.PointIdsList(points=ids))


class HybridDb:
    def __init__(self, path: str, embedding_function_local: bool = False) -> None:
        self.path = path
        self.client = QdrantClient(path=path)
        self.fts_store = FTSStore(os.path.join(path, "fts.db"))
        self.collections: Dict[str, Collection] = {}
        self.meta_store = CollectionMetaStore(path)
        self.embedding_function_local = embedding_function_local
        self._embedder: Optional[DenseEmbedder] = None
        self._sparse_encoder = SparseEncoder()

    def _get_embedder(self) -> DenseEmbedder:
        if self._embedder is None:
            if self.embedding_function_local:
                self._embedder = SentenceTransformerEmbedder()
            else:
                model = "qwen3-embedding-4b"
                api_base = os.getenv("PRIVATEMODE_API_BASE", "http://localhost:8080/v1")
                api_key = os.getenv("PRIVATEMODE_API_KEY", "dummy")
                logger.info("Using OpenAI embedding function with model '%s' and API base '%s'", model, api_base)
                self._embedder = OpenAIEmbedder(model=model, api_base=api_base, api_key=api_key)
        return self._embedder

    def _collection_exists(self, name: str) -> bool:
        collections = self.client.get_collections().collections
        return any(c.name == name for c in collections)

    def _ensure_collection(self, name: str, embed: bool, dim: int) -> None:
        if self._collection_exists(name):
            return

        if embed:
            try:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config={
                        "dense": VectorParams(size=dim, distance=Distance.COSINE),
                    },
                    sparse_vectors={
                        "sparse": SparseVectorParams(index=True),
                    },
                )
            except TypeError:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config={
                        "dense": VectorParams(size=dim, distance=Distance.COSINE),
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(index=True),
                    },
                )
        else:
            self.client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": VectorParams(size=1, distance=Distance.COSINE),
                },
            )

    def set_embedding_function(self, coll_name: str, embedding_fn: Optional[DenseEmbedder]) -> None:
        meta = self.meta_store.get(coll_name)
        has_embedding = embedding_fn is not None
        if meta:
            current_has = bool(meta.get("has_embedding", False))
            if current_has != has_embedding:
                raise ValueError(
                    f"Collection '{coll_name}' already exists with different embedding setting: {current_has}"
                )
            return

        self.meta_store.set(coll_name, {"has_embedding": has_embedding})

    def get_collection(self, name: str, constructor: type = Collection) -> Collection:
        meta = self.meta_store.get(name)
        if not meta:
            raise ValueError(f"Collection '{name}' does not exist")
        embed = bool(meta.get("has_embedding", False))
        if name not in self.collections:
            if not self._collection_exists(name):
                raise ValueError(f"Collection '{name}' does not exist")
            embedder = self._get_embedder() if embed else None
            self.collections[name] = constructor(
                name=name,
                client=self.client,
                fts_store=self.fts_store,
                embedder=embedder,
                sparse_encoder=self._sparse_encoder if embed else None,
                embed_enabled=embed,
            )
        return self.collections[name]

    def create_collection(self, name: str, embed: bool, constructor: type = Collection) -> Collection:
        embedder = self._get_embedder() if embed else None
        dim = embedder.dimension() if embedder else 1
        self._ensure_collection(name, embed, dim)
        self.set_embedding_function(name, embedder if embed else None)
        self.collections[name] = constructor(
            name=name,
            client=self.client,
            fts_store=self.fts_store,
            embedder=embedder,
            sparse_encoder=self._sparse_encoder if embed else None,
            embed_enabled=embed,
        )
        return self.collections[name]

    def reset(self) -> None:
        self.collections = {}
        collections = self.client.get_collections().collections
        for coll in collections:
            self.client.delete_collection(coll.name)
            self.meta_store.delete(coll.name)
