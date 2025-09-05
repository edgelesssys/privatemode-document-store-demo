"""
Simple chunk-level FTS storage using SQLite FTS5.

Designed for a single-node, on-device backend where documents are already
chunked for vector search. This module mirrors those chunks into a tiny
lexical index to enable BM25-style keyword queries and hybrid fusion.

API surface (stable):
- FTSStore(db_path)
    - init()                  -> ensure DB + schema + PRAGMAs
    - upsert_chunks(rows)     -> insert or replace by chunk_id
    - delete_doc(doc_id)      -> remove all chunks of a document
    - delete_chunks(ids)      -> remove specific chunk_ids
    - search(query, limit=20) -> list of hits with bm25 rank + normalized score
    - close()

Notes
-----
* Uses unicode61 tokenizer with diacritic removal. No stemming; add an
  auxiliary n-gram index or swap engines later if you need morphology.
* bm25() in SQLite returns lower-is-better ranks. We also return a
  simple normalized score = 1/(1+rank) for easy fusion.
* All operations are parameterized. MATCH accepts bound parameters in SQLite.
* WAL mode for concurrent readers; explicit batched transactions for writes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Any
import sqlite3
import threading
import json
import os


@dataclass(frozen=True)
class FTSChunk:
    chunk_id: str
    doc_id: str
    text: str
    title: Optional[str] = None
    tags: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None


class FTSStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        # Ensure parent dir exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        # check_same_thread=False to allow usage across threads with our own lock
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_pragmas()
        self._ensure_schema()

    # --- internals ---------------------------------------------------------
    def _init_pragmas(self) -> None:
        c = self._conn.cursor()
        # Reasonable defaults for local DBs
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("PRAGMA temp_store=MEMORY;")
        # ~200MB cache if negative (pages)
        c.execute("PRAGMA cache_size=-200000;")
        # mmap speeds up reads on 64-bit systems
        try:
            c.execute("PRAGMA mmap_size=30000000000;")
        except sqlite3.DatabaseError:
            pass
        c.close()

    def _ensure_schema(self) -> None:
        with self._conn:  # autocommit transaction
            # FTS5 table. Keep id fields UNINDEXED inside FTS and add a b-tree index for deletes.
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                    chunk_id UNINDEXED,
                    doc_id   UNINDEXED,
                    text,
                    title,
                    tags,
                    meta_json,
                    tokenize = 'unicode61 remove_diacritics 2'
                );
                """
            )
            # Note: FTS5 virtual tables cannot have regular indexes
            # Deletes will be slower but this is acceptable for this use case

    # --- public API --------------------------------------------------------
    def init(self) -> None:
        """No-op for compatibility; schema is ensured in __init__."""
        return

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_chunks(self, rows: Iterable[FTSChunk], batch_size: int = 1000) -> int:
        """Insert or replace chunks by chunk_id.

        Strategy: delete-by-chunk_id then insert (FTS5 has no real UPSERT).
        Returns number of inserted rows.
        """
        total = 0
        with self._lock:
            cur = self._conn.cursor()
            try:
                batch: List[FTSChunk] = []
                for r in rows:
                    batch.append(r)
                    if len(batch) >= batch_size:
                        total += self._write_batch(cur, batch)
                        batch.clear()
                if batch:
                    total += self._write_batch(cur, batch)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()
        return total

    def _write_batch(self, cur: sqlite3.Cursor, batch: List[FTSChunk]) -> int:
        # Delete existing chunk_ids (idempotent writes)
        cur.executemany(
            "DELETE FROM chunks WHERE chunk_id = ?;",
            [(r.chunk_id,) for r in batch],
        )
        # Insert fresh rows
        cur.executemany(
            """
            INSERT INTO chunks (chunk_id, doc_id, text, title, tags, meta_json)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    r.chunk_id,
                    r.doc_id,
                    r.text,
                    r.title or None,
                    ",".join(r.tags) if r.tags else None,
                    json.dumps(r.meta, ensure_ascii=False) if r.meta is not None else None,
                )
                for r in batch
            ],
        )
        return len(batch)

    def delete_doc(self, doc_id: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM chunks WHERE doc_id = ?;", (doc_id,))
            return cur.rowcount or 0

    def delete_chunks(self, chunk_ids: Iterable[str]) -> int:
        ids = list(chunk_ids)
        if not ids:
            return 0
        with self._lock, self._conn:
            q = "DELETE FROM chunks WHERE chunk_id = ?;"
            self._conn.executemany(q, [(i,) for i in ids])
            # sqlite3 doesn't reliably report total rowcount across executemany; return len(ids)
            return len(ids)

    def search(
        self,
        query: str,
        limit: int = 20,
        with_snippet: bool = True,
        snippet_tokens: int = 16,
        updated_after: Optional[float] = None,
        updated_before: Optional[float] = None,
        collection: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search over chunk text/title/tags.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            with_snippet: Whether to include highlighted snippets.
            snippet_tokens: Number of tokens around matches for snippets.
            updated_after: Filter chunks updated after this Unix timestamp (inclusive).
            updated_before: Filter chunks updated before this Unix timestamp (inclusive).
            collection: Optional collection name to limit search to (filters by doc_id prefix).

        Returns a list of dicts: {
            'chunk_id': str,
            'doc_id': str,
            'rank': float,              # bm25(chunks), lower is better
            'score': float,             # 1/(1+rank), higher is better
            'text': str,                # stored text
            'title': Optional[str],
            'tags': Optional[str],
            'meta': Optional[dict],
            'snippet': Optional[str],   # highlighted fragment (if requested)
        }
        """
        if not query or not query.strip():
            return []
        query = query.strip()

        # Columns: 0 chunk_id, 1 doc_id, 2 text, 3 title, 4 tags, 5 meta_json
        base_sql = (
            "SELECT chunk_id, doc_id, text, title, tags, meta_json, "
            "bm25(chunks) AS rank"
        )
        snip_sql = (
            ", snippet(chunks, 2, '<b>', '</b>', ' … ', ?) AS snippet"
            if with_snippet
            else ", NULL AS snippet"
        )
        where_clauses = ["chunks MATCH ?"]
        params: List[Any] = [query]
        if with_snippet:
            params.insert(0, snippet_tokens)
        if collection is not None:
            where_clauses.append("doc_id LIKE ?")
            params.append(f"{collection}-%")
        if updated_after is not None:
            where_clauses.append("json_extract(meta_json, '$.updated_at') > ?")
            params.append(updated_after)
        if updated_before is not None:
            where_clauses.append("json_extract(meta_json, '$.updated_at') < ?")
            params.append(updated_before)
        where_sql = " AND ".join(where_clauses)
        order_limit = " ORDER BY rank LIMIT ?;"
        sql = base_sql + snip_sql + " FROM chunks WHERE " + where_sql + order_limit
        params.append(limit)

        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute(sql, params)
                rows = cur.fetchall()
            finally:
                cur.close()

        results: List[Dict[str, Any]] = []
        for r in rows:
            rank = float(r[6])
            score = 1.0 / (1.0 + rank) if rank >= 0 else 0.0
            meta = None
            if r[5] is not None:
                try:
                    meta = json.loads(r[5])
                except Exception:
                    meta = None
            results.append(
                {
                    "chunk_id": r[0],
                    "doc_id": r[1],
                    "text": r[2],
                    "title": r[3],
                    "tags": r[4],
                    "meta": meta,
                    "rank": rank,
                    "score": score,
                    "snippet": r[7] if with_snippet else None,
                }
            )
        return results


# Convenience factory ---------------------------------------------------------

def open_store(db_path: str) -> FTSStore:
    """Open or create an FTS store at the given path."""
    return FTSStore(db_path)
