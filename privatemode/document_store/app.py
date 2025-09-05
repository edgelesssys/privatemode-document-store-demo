from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from privatemode.document_store.hybrid_db import Collection, HybridDb
from privatemode.document_store.auth import verify_auth, set_auth_key_path, set_allowed_origins


logger = logging.getLogger("privatemode.document_store")


def _setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = True
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )


_setup_logging()

def _resolve_embedding_model(embedding_function: object) -> str:
    for attr in ("model_name", "model", "name", "MODEL_NAME"):
        val = getattr(embedding_function, attr, None)
        if isinstance(val, str) and val:
            return val
    return embedding_function.__class__.__name__


vector_db_config = {
    "path": os.getenv("VECTOR_DB_PATH", os.path.join("data", "chroma")),
    "local_embedding": False
}

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting privatemode.document_store service")
    try:
        os.makedirs(vector_db_config["path"], exist_ok=True)
    except OSError:
        logger.error("Failed to create VECTOR_DB_PATH=%s", vector_db_config["path"])

    try:
        vector_db = HybridDb(vector_db_config["path"], embedding_function_local=vector_db_config["local_embedding"])
        app.state.vector_db = vector_db
        app.state.embedding_model = _resolve_embedding_model(vector_db.embedding_function)
        logger.info("Vector DB initialized at %s", vector_db_config["path"])
    except Exception as e:  # pragma: no cover - defensive
        app.state.vector_db = None
        app.state.embedding_model = None
        logger.warning("Vector DB init failed: %s", e)
    try:
        yield
    finally:
        logger.info("Shutting down privatemode.document_store service")

app = FastAPI(title="Privatemode Document Store", version="0.1.0", lifespan=lifespan)


_cors_env = os.getenv("CORS_ALLOW_ORIGINS")

# for debugging the frontend as regular webapp we allow localhost
_debug_allow_localhost = os.getenv("DEBUG_ALLOW_LOCALHOST", "0") in {"1", "true", "yes", "on"}
if _cors_env:
    _allowed_origins_list = [o.strip() for o in _cors_env.split(",") if o.strip()]
elif _debug_allow_localhost:
    _allowed_origins_list = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
else:
    _allowed_origins_list = []

logger.info("CORS allowed origins: %s", _allowed_origins_list)

set_auth_key_path(vector_db_config["path"])
set_allowed_origins(_allowed_origins_list)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins_list,
    allow_origin_regex=None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler for validation errors (422)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log validation errors with request details
    logger.warning(
        "Request validation failed for %s %s from %s. Validation errors: %s",
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
        exc.errors()
    )

    # Note: Request body logging for validation errors is not currently implemented
    # due to FastAPI consuming the body during parsing before validation

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "code": "VALIDATION_ERROR"}
    )

# Exception handler for HTTP exceptions (4xx responses)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Only log request body for 4xx errors
    if 400 <= exc.status_code < 500:
        logger.warning(
            "HTTP %d error for %s %s from %s: %s",
            exc.status_code,
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
            exc.detail
        )

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )

def get_db() -> HybridDb:
    db = getattr(app.state, "vector_db", None)
    if db is None:
        raise HTTPException(status_code=503, detail={"code": "DB_NOT_READY", "message": "Vector DB not initialized"})
    return db

@app.get("/health", tags=["system"])
async def health(request: Request) -> dict[str, str]:
    return {"status": "ok"}


class HybridWeights(BaseModel):
    bm25: Optional[float] = Field(default=None, ge=0, le=1)
    vector: Optional[float] = Field(default=None, ge=0, le=1)


class VectorSpec(BaseModel):
    embedding: Optional[List[float]] = None
    model: Optional[str] = None


class RerankSpec(BaseModel):
    enabled: Optional[bool] = None
    model: Optional[str] = None
    top_n: Optional[int] = None


class GroupBySpec(BaseModel):
    field: Optional[str] = None
    per_group: Optional[int] = None


class SpanSpec(BaseModel):
    window_tokens: Optional[int] = None
    surround_tokens: Optional[int] = None


class QueryRequest(BaseModel):
    query: Optional[str] = None
    top_k: int = Field(ge=1, le=100, default=10)
    hybrid: Optional[HybridWeights] = None
    vector: Optional[VectorSpec] = None
    filters: Optional[Dict[str, Any]] = None
    rerank: Optional[RerankSpec] = None
    group_by: Optional[GroupBySpec] = None
    span: Optional[SpanSpec] = None
    timeout_ms: Optional[int] = None
    collection: str = Field(..., min_length=1)
    trace: Optional[bool] = None


class Offset(BaseModel):
    start: int
    end: int


class Hit(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    raw_scores: Dict[str, float]
    text: str
    offset: Offset
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    took_ms: int
    hits: List[Hit]
    exhaustive: bool
    embedding_model: Optional[str] = None
    index_version: Optional[str] = None

class AdvancedQueryResponse(BaseModel):
    took_ms: int
    hits: List[Hit]
    history_content: List[Dict[str, Any]]
    history_summary: List[Dict[str, Any]]
    history_overview: List[Dict[str, Any]]
    exhaustive: bool
    embedding_model: Optional[str] = None
    index_version: Optional[str] = None

class AdvancedQueryRequest(BaseModel):
    messages: List[Dict[str, Any]]
    context_doc: Optional[str] = None
    top_k: int = Field(ge=1, le=100, default=10)
    collection: str = Field(..., min_length=1)


class StoreDocumentRequest(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_prefix: Optional[str] = None
    collection: str = Field(..., min_length=1)
    embed: Optional[bool] = True


class FullDocumentRequest(BaseModel):
    id: str
    text: str
    collection: str = Field(..., min_length=1)


def get_collection(collection: str, can_create: bool, embed: bool|None = None) -> Collection:
    if not collection:
        raise HTTPException(status_code=400, detail={"code": "MISSING_COLLECTION", "message": "collection is required"})

    db = get_db()

    try:
        return db.get_collection(collection)
    except ValueError:
        pass

    if not can_create or embed is None:
        raise HTTPException(status_code=404, detail={"code": "DB_ERROR", "message": "collection not found"})
    try:
        return db.create_collection(collection, embed=embed)
    except Exception as ce:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail={"code": "DB_ERROR", "message": str(ce)})

def get_meta_collection(coll_name: str) -> Optional[Collection]:
    """Get the collection to use for document listing and whether it has embeddings."""
    coll = get_collection(coll_name, can_create=False)
    has_embeddings = coll.collection._embedding_function is not None
    if not has_embeddings:
        return coll

    coll_meta_name = f"meta__{coll_name}"
    try:
        return get_collection(coll_meta_name, can_create=False)
    except HTTPException:
        return None

@app.post("/retrieval/query", response_model=QueryResponse, tags=["retrieval"])
async def retrieval_query(req: QueryRequest, request: Request) -> QueryResponse:
    verify_auth(request)
    logger.info("/retrieval/query called: top_k=%s collection=%s", req.top_k, req.collection)
    if not req.query:
        raise HTTPException(status_code=400, detail={"code": "MISSING_QUERY", "message": "query is required"})
    coll = get_collection(req.collection, can_create=False)
    t0 = time.perf_counter()
    try:
        res = coll.search(
            req.query,
            max_results=req.top_k,
            include=["distances", "metadatas", "documents"],
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Vector search failed")
        raise HTTPException(status_code=500, detail={"code": "SEARCH_ERROR", "message": str(e)})
    took_ms = int((time.perf_counter() - t0) * 1000)
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    hits: List[Hit] = []
    for i, chunk_id in enumerate(ids):
        text = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        dist = float(dists[i]) if i < len(dists) else 1.0
        score = max(0.0, min(1.0, 1.0 - dist / 4.0))
        base_id = chunk_id.rsplit("-", 1)[0] if "-" in chunk_id else chunk_id
        hits.append(
            Hit(
                doc_id=base_id,
                chunk_id=chunk_id,
                score=score,
                raw_scores={"vector": score},
                text=text or "",
                offset=Offset(start=0, end=len(text or "")),
                metadata=meta,
            )
        )
    embed_model = getattr(app.state, "embedding_model", None)
    return QueryResponse(
        took_ms=took_ms,
        hits=hits,
        exhaustive=False,
        embedding_model=embed_model or "unknown-embedding-model",
        index_version="chroma-v1",
    )


@app.post("/retrieval/query-advanced", response_model=AdvancedQueryResponse, tags=["retrieval"])
async def retrieval_query_advanced(req: AdvancedQueryRequest, request: Request) -> AdvancedQueryResponse:
    from privatemode.document_store.query_advanced import run_query_advanced
    verify_auth(request)
    if not req.messages:
        raise HTTPException(status_code=400, detail={"code": "MISSING_MESSAGES", "message": "messages is required and cannot be empty"})
    chunks_coll = get_collection(req.collection, can_create=False)
    documents_coll = get_meta_collection(req.collection) or chunks_coll  # fallback to chunks_coll if no meta collection
    try:
        result = await run_query_advanced(req.messages, documents_coll, chunks_coll, req.top_k, req.context_doc)
    except Exception as e:
        logger.exception("Vector search failed")
        raise HTTPException(status_code=500, detail={"code": "SEARCH_ERROR", "message": str(e)})
    embed_model = getattr(app.state, "embedding_model", None)
    history=result["history"]
    return AdvancedQueryResponse(
        took_ms=result["took_ms"],
        hits=result["hits"],
        history_content=history["browser_history_content"],
        history_summary=history["browser_history_summary"],
        history_overview=history["browser_history_overview"],
        exhaustive=False,
        embedding_model=embed_model or "unknown-embedding-model",
        index_version="chroma-v1",
    )


@app.post("/documents", tags=["retrieval"], status_code=201)
async def store_document(req: StoreDocumentRequest, request: Request) -> Dict[str, Any]:
    verify_auth(request)
    embed: bool = req.embed if req.embed is not None else True
    coll = get_collection(req.collection, can_create=True, embed=embed)
    try:
        req.metadata["updated_at"] = datetime.now().isoformat()
        if embed:
            n = coll.upsert(req.id, req.text, req.metadata, chunk_prefix=req.chunk_prefix)
            if n > 0:
                # additionally store a no-embed summary record for title/metadata-only retrieval
                coll_meta_name = f"meta__{req.collection}"
                coll_meta = get_collection(coll_meta_name, can_create=True, embed=False)
                coll_meta.upsert_noembed(req.id, req.text, req.metadata)
        else:
            n = coll.upsert_noembed(req.id, req.text, req.metadata)

            # for no-embed collections, we currently do not store a meta collection
            # as it can be fetched from the full document

        return {"status": "ok", "id": req.id, "chunks": n}
    except Exception as e:
        logger.exception("Failed to store document", exc_info=e)
        if "Use upsert() for collections with embeddings" in str(e):
            raise HTTPException(
                status_code=400,
                detail={"code": "STORE_ERROR", "message": "The collection has embeddings but you requested storage without embeddings"}
            )
        raise HTTPException(status_code=500, detail={"code": "STORE_ERROR", "message": str(e)})


@app.post("/admin/reset", tags=["admin"])
async def reset_database(request: Request) -> Dict[str, Any]:
    verify_auth(request)
    try:
        get_db().reset()
        return {"status": "ok"}
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Failed to reset database")
        raise HTTPException(status_code=500, detail={"code": "RESET_ERROR", "message": str(e)})

@app.get("/documents/{collection}/{doc_id}", tags=["retrieval"])
async def get_document(collection: str, doc_id: str, request: Request) -> Dict[str, Any]:
    """Return the full stored document (if present) as JSON."""
    verify_auth(request)
    coll = get_collection(collection, can_create=False)
    try:
        res = coll.get(doc_id)
        docs = res.get("documents", [])
        if not docs:
            raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Document not found"})
        return {
            "status": "ok",
            "id": doc_id,
            "collection": collection,
            "docs": docs,
            "size": sum(len(doc) for doc in docs),
            "metadata": res.get("metadatas", []),
        }
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Failed to read full document")
        raise HTTPException(status_code=500, detail={"code": "FULLDOC_READ_ERROR", "message": str(e)})


@app.delete("/documents/{collection}/{doc_id}", tags=["retrieval"])
async def delete_document(collection: str, doc_id: str, request: Request) -> Response:
    verify_auth(request)
    coll = get_collection(collection, can_create=False)
    try:
        coll.delete(doc_id)
        # Also delete from meta collection
        coll_meta_name = f"meta__{collection}"
        try:
            coll_meta = get_collection(coll_meta_name, can_create=False, embed=False)
            coll_meta.delete(doc_id)
        except HTTPException:
            # If meta collection doesn't exist, ignore
            pass
        return Response(status_code=204)
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Failed to delete full document")
        raise HTTPException(status_code=500, detail={"code": "FULLDOC_DELETE_ERROR", "message": str(e)})


@app.get("/documents/{collection}", tags=["retrieval"])
async def list_documents(collection: str, request: Request, limit: Optional[int] = Query(None, ge=1, le=100)) -> Dict[str, Any]:
    verify_auth(request)
    meta_coll = get_meta_collection(collection)
    if meta_coll is None:
        return {"documents": []}

    try:
        # only fetch metadatas and ids, not full documents
        res = meta_coll.collection.get(include=["metadatas"])
        metas = res.get("metadatas", [])
        ids = res.get("ids", [])
        documents = []
        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            # For non-embedded, title might be in metadata, otherwise use doc_id
            title = meta.get("title", doc_id)
            documents.append({
                "id": doc_id,
                "title": title,
                "metadata": meta
            })
        # Sort documents by updated_at descending (newest first)
        documents.sort(key=lambda x: x.get("metadata", {}).get("updated_at", ""), reverse=True)
        # Apply limit if specified
        if limit:
            documents = documents[:limit]
        return {"documents": documents}
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Failed to list documents")
        raise HTTPException(status_code=500, detail={"code": "LIST_ERROR", "message": str(e)})
