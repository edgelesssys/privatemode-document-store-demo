#!/usr/bin/env bash
set -euo pipefail

# Fetch a full (unchunked) stored document via /full_documents/{collection}/{id}.
# Usage:
#   BASE_URL=http://localhost:8081 ./tests/smoke/full_documents/get.sh

BASE_URL=${BASE_URL:-http://localhost:8081}
COLLECTION=${COLLECTION:-test-full-docs}
DOC_ID=${DOC_ID:-fulldoc-demo}

url="$BASE_URL/documents/$COLLECTION/$DOC_ID"
echo "[smoke] GET /documents/{collection}/{id} -> $url"
status=$(curl -sS -w '%{http_code}' -o /tmp/full_doc_get_resp.json "$url")
echo "Status: $status"
jq . /tmp/full_doc_get_resp.json || cat /tmp/full_doc_get_resp.json

if [[ "$status" != "200" ]]; then
  echo "Get full document failed" >&2
  exit 1
fi
