#!/usr/bin/env bash
set -euo pipefail

# Store a full (unchunked) document via the /full_documents API and print the response.
# Usage:
#   BASE_URL=http://localhost:8081 ./tests/smoke/full_documents/store.sh

BASE_URL=${BASE_URL:-http://localhost:8081}
COLLECTION=${COLLECTION:-test-full-docs}
DOC_ID=${DOC_ID:-fulldoc-demo}

payload=$(cat <<JSON
{
  "id": "${DOC_ID}",
  "collection": "${COLLECTION}",
  "text": "This is a demonstration full document body stored at $(date -u +%Y-%m-%dT%H:%M:%SZ).",
  "embed": false
}
JSON
)

echo "[smoke] POST /documents -> $BASE_URL/documents"
status=$(curl -sS -w '%{http_code}' -o /tmp/full_doc_store_resp.json \
  -X POST "$BASE_URL/documents" \
  -H 'Content-Type: application/json' \
  -d "$payload")

jq . /tmp/full_doc_store_resp.json || cat /tmp/full_doc_store_resp.json
echo "Status: $status"

if [[ "$status" != "201" ]]; then
  echo "Store full document failed" >&2
  exit 1
fi
