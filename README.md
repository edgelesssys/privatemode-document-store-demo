# Privatemode Document Store (Experimental)

This is a demo of document storage and retrieval powered by the embeddings and LLM inference services of [Privatemode AI](https://www.privatemode.ai/). The system combines a local ChromaDB vector store with SQLite-based keyword search (BM25) to enable hybrid retrieval. All data remains encrypted end-to-end during embedding and LLM inference in the cloud, ensuring privacy-preserving AI operations.

## 🚀 Getting started

### Prerequisites 

Make sure the [Privatemode Proxy](https://docs.privatemode.ai/guides/proxy-configuration) is running. Enable [prompt caching](https://docs.privatemode.ai/guides/proxy-configuration#prompt-caching) in the proxy to reduce latency and cost.

### macOS and Linux

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

## 🛠️ Development

For installation as background service see [INSTALL.md](INSTALL.md).

### Components

- FastAPI app (`privatemode.document_store`), served by Uvicorn
- REST API:
  - GET `/health`
  - POST `/documents` (store; auto-chunking, stable chunk IDs, metadata with hash and chunk count)
  - POST `/retrieval/query` (vector search; returns text and scores)
  - POST `/admin/reset` (wipe data)
- ChromaDB as vector store
- Sqlite for keyword search (bm25)

### Environment variables

The following variables can be set optionally:

- HOST (default 127.0.0.1)
- PORT (default 8081)
- RELOAD (0/1; default 0)
- LOG_LEVEL (debug,info,warning,error; default info)
- PRIVATEMODE_API_KEY (default: none; i.e., the value configured in the Privatemode proxy)
- PRIVATEMODE_API_BASE (default: `http://localhost:8080/v1`)
- VECTOR_DB_PATH (default: ./data/chroma - relative to working directory)

### Smoke test

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

### Running tests

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
