# Retrieval API

This document defines the minimal HTTP contract for the storage service.

## Endpoint

POST /retrieval/query

- Purpose: retrieve top passages for a query with optional filters, hybrid weighting, and reranking.
- Security:
  - auth TBD (e.g., bearer token).
  - `tenant` must be enforced both at index and query time; currently not implemented
  - encryption with user keys required; currently not implemented

## Request (canonical fields)

```jsonc
{
  "query": "How does backpressure work in Kafka?",   // required if no vector.embedding
  "top_k": 12,                                        // [1..100]
  "hybrid": { "bm25": 0.4, "vector": 0.6 },         // optional; weights sum ~1
  "vector": { "embedding": [/* optional */], "model": "e5-large-v2" },
  "filters": {                                        // AND of fields; OR via {"any": [...]}
    "any": [
      { "eq": { "namespace": "docs" } },
      { "eq": { "namespace": "blog" } }
    ],
    "gte": { "published_at": "2022-01-01" },
    "in":  { "lang": ["en", "de"] },
    "contains_any": { "tags": ["streaming", "throughput"] }
  },
  "rerank": { "enabled": true, "model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "top_n": 8 },
  "group_by": { "field": "doc_id", "per_group": 3 },
  "span": { "window_tokens": 180, "surround_tokens": 40 },
  "timeout_ms": 800,
  "collection": "my-docs",
  "tenant": "acme",
  "trace": false
}
```

Notes:

- query/vector: support text and client-supplied embeddings; server embeds if none provided.
- hybrid: standard BM25 + ANN blend; weights tune lexical vs. semantic.
- filters: simple JSON DSL (AND/OR, range, set ops) that works across engines.
- rerank: cross-encoder on a shortlist keeps search fast but high quality.
- group_by: diversity to avoid many chunks from the same document.
- span: tight snippets aligned to token windows.

## Response (canonical shape)

```jsonc
{
  "took_ms": 73,
  "hits": [
    {
      "doc_id": "kb:123e4567",
      "chunk_id": "kb:123e4567#c07",
      "score": 38.21,
      "raw_scores": { "bm25": 7.2, "vector": 0.84, "rerank": 0.92 },
      "text": "Kafka uses backpressure by ...",
      "offset": { "start": 5210, "end": 5590 },
      "metadata": {
        "title": "Operating Kafka at Scale",
        "namespace": "docs",
        "url": "https://…/kafka.html",
        "published_at": "2023-03-18",
        "lang": "en",
        "tags": ["kafka", "throughput", "backpressure"]
      }
    }
  ],
  "exhaustive": false,
  "embedding_model": "e5-large-v2@2025-07-12",
  "index_version": "docs-v4-2025-09-01"
}
```

Why these fields?

- IDs & versions: allow caching, dedupe, auditing, reproducibility.
- collection: separate collections with different types of documents
- tenant (TBD): clear isolation boundary (beyond namespaces).

## Errors

```json
{ "error": { "code": "TIMEOUT", "message": "Reranker exceeded 800ms" } }
```

Common codes: BAD_REQUEST, UNAUTHORIZED, FORBIDDEN, TIMEOUT, INDEX_NOT_READY, EMBED_MODEL_MISMATCH.

## Optional endpoints (nice to have)

- POST /retrieval/mget — fetch by doc_id/chunk_id.
- POST /retrieval/expand — query expansion (synonyms/paraphrases).
- POST /retrieval/feedback — labels for online tuning.
- GET  /retrieval/stats — latency/recall histograms, index & model versions.
- POST /retrieval/ablation — run A/B on saved queries, return leaderboard.

## Minimal OpenAPI fragment

```yaml
paths:
  /retrieval/query:
    post:
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RetrievalQuery'
      responses:
        '200':
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RetrievalResponse'
components:
  schemas:
    RetrievalQuery:
      type: object
      required: [query, top_k]
      properties:
        query: { type: string }
        top_k: { type: integer, minimum: 1, maximum: 100 }
        hybrid:
          type: object
          properties:
            bm25:  { type: number, minimum: 0, maximum: 1 }
            vector:{ type: number, minimum: 0, maximum: 1 }
        vector:
          type: object
          properties:
            embedding: { type: array, items: { type: number } }
            model: { type: string }
        filters: { type: object, additionalProperties: true }
        rerank:
          type: object
          properties:
            enabled: { type: boolean }
            model: { type: string }
            top_n: { type: integer }
        group_by:
          type: object
          properties:
            field: { type: string }
            per_group: { type: integer }
        span:
          type: object
          properties:
            window_tokens: { type: integer }
            surround_tokens: { type: integer }
        timeout_ms: { type: integer }
        collection: { type: string }
        tenant: { type: string }
        trace: { type: boolean }
    RetrievalResponse:
      type: object
      properties:
        took_ms: { type: integer }
        hits:
          type: array
          items:
            type: object
            properties:
              doc_id: { type: string }
              chunk_id: { type: string }
              score: { type: number }
              raw_scores: { type: object, additionalProperties: { type: number } }
              text: { type: string }
              offset:
                type: object
                properties: { start: { type: integer }, end: { type: integer } }
              metadata: { type: object, additionalProperties: true }
        exhaustive: { type: boolean }
        embedding_model: { type: string }
        index_version: { type: string }
```

Implementation notes (abridged): hybrid first, rerank second; MMR/diversity via group_by; deterministic IDs; version embeddings and index; consider a POST /retrieval/pack for context packing.
