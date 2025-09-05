
import time
import asyncio
import os
import re
from typing import List, Dict, Any, Optional, cast
from urllib.parse import urlparse
from privatemode.document_store.hybrid_db import Collection
from .app import Hit, Offset
import openai
import logging

logger = logging.getLogger("privatemode.document_store")

def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc or parsed.path or "unknown"
    except:
        return "unknown"

def filter_messages(messages: List[Dict[str, Any]], k: int) -> List[str]:
    filtered = []
    arr = messages if isinstance(messages, list) else []
    for i in range(len(arr) - 1, -1, -1):
        if len(filtered) >= k:
            break
        msg = arr[i]
        content = msg.get("content", "")
        if msg.get("role") != "system" and content and isinstance(content, str) and content.strip():
            filtered.append(content.strip())
    return filtered

def hash_text(s: str) -> str:
    s = str(s or "")
    h1 = 0x811c9dc5
    for c in s:
        h1 ^= ord(c)
        h1 = (h1 * 0x01000193) & 0xFFFFFFFF
    h1 ^= h1 >> 13; h1 = (h1 * 0x5bd1e995) & 0xFFFFFFFF; h1 ^= h1 >> 15
    return format(h1 & 0xFFFFFFFF, "08x")

def norm_text(s: str) -> str:
    return str(s or "").strip().lower()

def rrf_fuse(sem_results: List[List[Hit]], weights: List[float], kw_results: List[List[Hit]], rrf_k: Optional[int] = None) -> List[Hit]:
    if rrf_k is None:
        lens = [len(r) for r in sem_results + kw_results if isinstance(r, list) and len(r) > 0]
        avg = (sum(lens) / len(lens)) if lens else 5
        rrf_k = max(10, min(60, round(avg * 3)))

    fused_map = {}
    for idx, hits in enumerate(sem_results):
        w = weights[idx] if idx < len(weights) else 1.0
        for rank_idx, h in enumerate(hits):
            key = "t#" + hash_text(norm_text(h.text))
            add = w * (1 / (rrf_k + (rank_idx + 1)))
            if key not in fused_map:
                # Clone the hit to avoid mutating original
                hit = h.model_copy(update={"score": add}) if hasattr(h, "model_copy") else h
                fused_map[key] = {"hit": hit, "rrf": add}
            else:
                fused_map[key]["rrf"] += add

    # Fuse keyword results with weight 1.0
    kw_weight = 1.0
    for idx, hits in enumerate(kw_results):
        w = kw_weight
        for rank_idx, h in enumerate(hits):
            key = "t#" + hash_text(norm_text(h.text))
            add = w * (1 / (rrf_k + (rank_idx + 1)))
            if key not in fused_map:
                # Clone the hit to avoid mutating original
                hit = h.model_copy(update={"score": add}) if hasattr(h, "model_copy") else h
                fused_map[key] = {"hit": hit, "rrf": add}
            else:
                fused_map[key]["rrf"] += add

    fused = [ (f["hit"], f["rrf"]) for f in fused_map.values() ]
    fused.sort(key=lambda x: x[1], reverse=True)
    # Update score field in Hit
    result = []
    for hit, score in fused:
        if hasattr(hit, "model_copy"):
            hit = hit.model_copy(update={"score": score})
        else:
            hit.score = score
        result.append(hit)
    return result

async def retrieve_one(query: str, collection: Collection, top_k: int) -> List[Hit]:
    res = collection.search(
        query,
        max_results=top_k,
        include=["distances", "metadatas", "documents"],
    )

    # TODO: if the query was split into multiple chunks, merge results from different chunks instead of using the first only
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
    return hits

async def llm_filter_query(context_doc: str, messages: List[str], last_question: str, browser_history_overview: List[Dict[str, Any]]) -> str:
    api_key = os.getenv("PRIVATEMODE_API_KEY", "dummy")
    base_url = os.getenv("PRIVATEMODE_API_BASE", "http://localhost:8080/v1")
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    page_content = (context_doc or "")[:1000]
    conversation = "\n".join(messages)
    history_overview = "\n".join(f"- {idx}: {extract_domain(item.get('metadata', {}).get('url', ''))} - {item.get('title', 'Unknown')}" for idx, item in enumerate(browser_history_overview[:20]))  # Limit to 20 for prompt length
    system_prompt = "You are an AI assistant in the browser that helps users by providing relevant information based on the provided context, conversation history, and browser history."
    prompt = f'''Current website content:\n{page_content}

Conversation history:
{conversation}

Browser history overview (recent titles and URLs):
{history_overview}

User question: {last_question}

Do not answer the question, just tell me what information do you need to answer this question?
- Content or summary of recently visited websites (content of the current page is always available)
- What keywords should be used to search the browser history
- What time range (in hours) should be used to filter the browser history (e.g. last 24 hours, last 48 hours, etc.)

Return a json object:
{{
  "content_full": [int],          // index of pages you need the full content from (at most 2 pages)
  "content_summary": [int],       // index of pages you need a short summary from (at most 3 pages)
  "keywords": [string],           // list of keywords/search strings to search in browser history
  "history_filter_hours": [int],  // start and end of history (hours in the past from now) for semantic and keyword search (e.g. [0, 24] for last day or [24, 48] for previous day or [], if no filtering)
}}

The current time is {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.
'''

    try:
        t_llm_start = time.perf_counter()
        response = await client.chat.completions.create(
            model="qwen3-coder-30b-a3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        t_llm_end = time.perf_counter()
        llm_took_ms = int((t_llm_end - t_llm_start) * 1000)
        logger.info(f"LLM query took {llm_took_ms} ms")
        content = response.choices[0].message.content
        return content if content else "No response"
    except Exception as e:
        logger.error(f"LLM Error: {str(e)}")
        raise

def parse_llm_response(s: str) -> dict[str, Any]:
    res: dict[str, Any] = {}
    lines = s.split('\n')
    for line in lines:
        line = line.strip().strip(",").strip()
        if '[' not in line:
            continue
        key_part, value_part = line.split('[', 1)
        key = key_part.strip().strip(":").strip('"').strip("'")
        if key not in ['content_full', 'content_summary', 'keywords', 'history_filter_hours']:
            continue
        value_part = value_part.strip().strip(']')
        vals = value_part.split(',')
        parsed = [v.strip() for v in vals if v.strip()]
        parsed = [int(v) if v.isdigit() else v for v in parsed]
        res[key] = parsed if key != 'keywords' else re.findall(r'"([^"]*)"', value_part)
    return res

def search_browser_history(collection: Collection, queries: List[str], history_filter_hours: List[int], limit: int) -> List[Hit]:
    # Calculate time filters
    updated_after = None
    updated_before = None
    if len(history_filter_hours) == 2:
        start_hours, end_hours = sorted(history_filter_hours)  # ensure start < end
        current_time = time.time()
        updated_after = current_time - (end_hours * 3600)
        updated_before = current_time - (start_hours * 3600)

    # Use dict to deduplicate and count frequency
    result_map = {}  # (doc_id, chunk_id) -> (hit, frequency)

    for query in queries:
        logger.info(f"Searching browser history for query: {query}")
        try:
            fts_results = collection.search_fts(
                query=query,
                limit=limit,  # use the provided limit
                updated_after=updated_after,
                updated_before=updated_before,
                with_snippet=True
            )
            for result in fts_results:
                key = (result['doc_id'], result['chunk_id'])
                hit = Hit(
                    doc_id=result['doc_id'],
                    chunk_id=result['chunk_id'],
                    score=result['score'],
                    raw_scores={'fts': result['score']},
                    text=result['text'],
                    offset=Offset(start=0, end=len(result['text'])),
                    metadata=result['meta'] or {}
                )
                if key in result_map:
                    # Increment frequency, keep the hit with highest score
                    existing_hit, freq = result_map[key]
                    if hit.score > existing_hit.score:
                        result_map[key] = (hit, freq + 1)
                    else:
                        result_map[key] = (existing_hit, freq + 1)
                else:
                    result_map[key] = (hit, 1)
        except Exception as e:
            logger.error(f"Error searching FTS for query '{query}': {e}")

    # Sort by frequency (descending), then by score (descending)
    sorted_results = sorted(result_map.values(), key=lambda x: (-x[1], -x[0].score))

    # Return just the hits, limited to the requested limit
    return [hit for hit, freq in sorted_results[:limit]]

def fetch_page_content(documents_collection: Collection, browser_history_overview: List[Dict[str, Any]], indices: List[int], full_content: bool = False) -> List[Dict[str, str]]:
    content_results = []
    max_chars = 20000 if full_content else 5000  # shorter for summaries

    if len(indices) > 1:
        max_chars = max_chars // len(indices)  # split if multiple full contents requested

    for idx in indices:
        if idx < 0 or idx >= len(browser_history_overview):
            logger.warning(f"Invalid index {idx} for browser history overview")
            continue

        doc_info = browser_history_overview[idx]
        doc_id = doc_info.get("id")  # url field contains the document ID

        logger.info(f"Fetching content for document {doc_info.get('title')} (index {idx})")

        try:
            # Get the full document content
            res = documents_collection.get(doc_id)
            docs = res.get("documents", [])
            metas = res.get("metadatas", [])

            if not docs or len(docs) != 1:
                logger.warning(f"Expected exactly one document for {doc_id}, got {len(docs)}")
                continue

            content = docs[0]  # There should be only one document
            meta = metas[0] if metas and isinstance(metas[0], dict) else {}

            # For summary, truncate if too long
            if len(content) > max_chars:
                content = content[:max_chars] + "..."

            content_results.append({
                "title": doc_info.get("title", "Unknown"),
                "url": doc_id,
                "content": content,
                "metadata": meta
            })

        except Exception as e:
            logger.error(f"Error fetching content for document {doc_id}: {e}")
            continue

    return content_results

def get_browse_history(collection: Collection, limit: Optional[int] = None, temporal_filter: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    res = collection.get(id=None)
    metas = res.get("metadatas", [])
    ids = res.get("ids", [])
    documents = []
    for i, doc_id in enumerate(ids):
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        # For non-embedded, title might be in metadata, otherwise use doc_id
        title = meta.get("title", doc_id)
        documents.append({
            "title": title,
            "id": doc_id,
            "metadata": meta
        })

    # Apply temporal filtering if valid
    if temporal_filter is not None:
        if (isinstance(temporal_filter, list) and len(temporal_filter) == 2 and
            all(isinstance(h, int) and h > 0 for h in temporal_filter)):
            start_hours = min(temporal_filter)
            end_hours = max(temporal_filter)
            current_time = time.time()
            start_time = current_time - (end_hours * 3600)  # older boundary
            end_time = current_time - (start_hours * 3600)   # newer boundary

            filtered_documents = []
            for doc in documents:
                updated_at = doc.get("metadata", {}).get("updated_at")
                if updated_at is not None and start_time <= updated_at <= end_time:
                    filtered_documents.append(doc)
            documents = filtered_documents
        else:
            logger.error(f"Invalid temporal_filter: {temporal_filter}. Expected list of 2 positive integers. Applying no filtering.")

    # Sort documents by updated_at descending (newest first)
    documents.sort(key=lambda x: x.get("metadata", {}).get("updated_at", ""), reverse=True)
    # Apply limit if specified
    if limit:
        documents = documents[:limit]
    return documents

async def fetch_advanced(documents_collection: Collection, chunks_collection: Collection, context_doc: str, messages: List[str], last_question: str) -> dict:
    browser_history_overview = get_browse_history(documents_collection, limit=20)

    res = await llm_filter_query(context_doc, messages, last_question, browser_history_overview)
    # parse JSON
    parsed_res = parse_llm_response(res)
    logger.info(f"LLM response parsed: {parsed_res}")
    history_filter_hours = parsed_res.get('history_filter_hours', [])

    # keyword query if needed
    keyword_results = []
    if parsed_res['keywords']:
        keyword_results = search_browser_history(chunks_collection, parsed_res['keywords'], history_filter_hours, limit=20)

    # fetch browser history
    ret_browser_history_content = fetch_page_content(documents_collection, browser_history_overview, parsed_res.get('content_full', []), full_content=True)
    ret_browser_history_summary = fetch_page_content(documents_collection, browser_history_overview, parsed_res.get('content_summary', []), full_content=False)

    # return browser history overview, filtered by time if needed
    ret_browser_history_overview = browser_history_overview

    # TODO: temporal filtering doesn't work well as the model usually adds a filter that excludes history without good reason
    # if history_filter_hours:
    #     browser_history_filtered = get_browse_history(documents_collection, limit=20, temporal_filter=history_filter_hours)
    #     if browser_history_filtered:
    #         ret_browser_history_overview = browser_history_filtered

    # TODO:
    # - let the model decide if we need history items at all
    # - let the model decide if we need current page content and if not,
    #   return a summary instead to reduce tokens and make space for other data
    # - reduce other content if one type of content is requested

    return {
        "keyword_search_results": keyword_results,   # list of Hit objects
        "browser_history_content": ret_browser_history_content,  # list of {title, url, content}
        "browser_history_summary": ret_browser_history_summary,  # list of {title, url, content}
        "browser_history_overview": ret_browser_history_overview, # list of {title, url}
    }

async def run_query_advanced(messages: List[Dict[str, Any]], documents_collection: Collection, chunks_collection: Collection, top_k: int, context_doc: Optional[str] = None) -> Dict[str, Any]:
    weights_reversed = [1.0, 0.85, 0.7, 0.55]
    k = len(weights_reversed)
    filtered_reversed = filter_messages(messages, k)
    if not filtered_reversed:
        return {"hits": [], "took_ms": 0, "llm_response": ""}
    t0 = time.perf_counter()

    # Run retrievals and LLM in parallel
    queries = filtered_reversed
    tasks = [retrieve_one(q, chunks_collection, top_k * 2) for q in queries]

    last_question = filtered_reversed[0]
    llm_task = fetch_advanced(documents_collection, chunks_collection, context_doc or "", list(reversed(filtered_reversed[1:])), last_question)

    all_tasks = tasks + [llm_task]
    results = await asyncio.gather(*all_tasks)
    took_ms = int((time.perf_counter() - t0) * 1000)
    retrieval_results = cast(List[List[Hit]], results[:-1])
    llm_result = cast(Dict[str, Any], results[-1])
    keyword_results = llm_result.get("keyword_search_results", [])

    # RRF fusion:
    # - semantic search results with per-query weights
    # - keyword search results with weight 1.0
    fused = rrf_fuse(retrieval_results, weights_reversed, [keyword_results])
    capped = fused[:top_k]
    return {
        "hits": capped,
        "took_ms": took_ms,
        "history": llm_result
    }
