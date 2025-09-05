# Privatemode Document Store (Experimental)

> **⚠️ EXPERIMENTAL - NOT FOR PRODUCTION USE**
>
> This service is an **experimental proof-of-concept** designed to validate the feasibility of confidential retrieval. It is **not production-ready** and should be used for exploration and demonstration purposes only.
>
> **What to expect:**
> - Bugs and limited functionality
> - No guarantees on maintenance or support
> - May be modified or removed at any time
> - Local storage is not production-grade: Browsing history stored locally without encryption
>
> **This is a vision piece** to show what's possible with privacy-preserving embeddings and inferecence - not something you should depend on for real-world use.
>
> **Note:** The Privatemode proxy itself provides confidential AI communication as designed - the experimental nature applies to this extension's implementation.

## Overview

This is a demonstration of document storage and retrieval powered by [Privatemode's](https://www.privatemode.ai) confidential embeddings and LLM inference services. The system combines a local ChromaDB vector store with SQLite-based keyword search (BM25) to enable hybrid retrieval. All data remains encrypted end-to-end during embedding and LLM inference in the cloud, ensuring privacy-preserving AI operations.

### 🔒 About the Privatemode Proxy

This service requires the [Privatemode proxy](https://docs.privatemode.ai/quickstart) to be running locally or on a trusted host. With Privatemode, your data and prompts are encrypted during processing and cannot be accessed by anyone but you nor can it be used for model training.

The Privatemode proxy is a lightweight service that does the following:

- It encrypts data sent to Privatemode and decrypts all data received.
- It verifies the integrity of the Privatemode backend.
- It exposes an OpenAI-compatible API endpoint for AI inference.

Run it via Docker:

```bash
docker run -p 8080:8080 \
  ghcr.io/edgelesssys/privatemode/privatemode-proxy:latest \
  --apiKey <your-api-key>
```

You can get started for free with a [Privatemode API key](https://www.privatemode.ai/pricing).

Learn more about Privatemode and the proxy in the [docs](https://docs.privatemode.ai/quickstart).

### 📦 Components
- FastAPI app (`privatemode.document_store`), served by Uvicorn
- REST API:
  - GET `/health`
  - POST `/documents` (store; auto-chunking, stable chunk IDs, metadata with hash and chunk count)
  - POST `/retrieval/query` (vector search; returns text and scores)
  - POST `/admin/reset` (wipe data)
- ChromaDB as vector store
- Sqlite for keyword search (bm25)

## Development

> For installation as background service see INSTALL.md.

- Python 3.12+ is required. Set up a virtual environment and install dependencies as described below.
- [Privatemode proxy](https://docs.privatemode.ai/guides/proxy-configuration) running on <http://localhost:8080> (default) or other host/port as configured. Make sure to enable [prompt caching](https://docs.privatemode.ai/guides/proxy-configuration#prompt-caching) in the proxy to reduce latency and cost.

### macOS/Linux

Install the venv using your favorite method or the provided script:

```bash
source scripts/venv.rc
```

Or with your own env setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

After installation, you can run it locally:

```bash
python -m privatemode.document_store
# Optional env overrides:
# HOST=0.0.0.0 PORT=8081 RELOAD=0 LOG_LEVEL=info python -m privatemode.document_store
```

Environment variables (optional):

- HOST (default 127.0.0.1)
- PORT (default 8081)
- RELOAD (0/1; default 0)
- LOG_LEVEL (debug,info,warning,error; default info)
- PRIVATEMODE_API_KEY (default: none; i.e., the value configured in the Privatemode proxy)
- PRIVATEMODE_API_BASE (default: `http://localhost:8080/v1`)
- VECTOR_DB_PATH (default: ./data/chroma - relative to working directory)

You can smoke test a running instance:

```bash
# Health
curl -sf http://127.0.0.1:8081/health && echo

# Store a document
curl -sS -X POST "http://127.0.0.1:8081/documents" \
  -H 'Content-Type: application/json' \
  -d '{"collection":"docs","id":"readme-smoke","text":"hello world","metadata":{}}'

# Query
curl -sS -X POST "http://127.0.0.1:8081/retrieval/query" \
  -H 'Content-Type: application/json' \
  -d '{"collection":"docs","query":"hello world","top_k":5}'
```

### Windows (PowerShell)

Install Python 3.12 or later from https://www.python.org/downloads/windows/ or Microsoft Store.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .[dev]
```

To run in foreground (dev/testing):

```bash
python -m privatemode.document_store
```

---

## Running tests

Run tests with pytest:

```bash
pytest
pytest -vv -s
```

The `-vv` flag increases verbosity to show detailed test output.
The `-s` flag enables capturing of stdout/stderr, showing print statements and logging in real-time.

Run a single test file or test by name:

```bash
pytest tests/test_health.py
pytest -k health
```

Useful flags:

- `-x` stop on first failure
- `-lf` rerun only last failed tests
