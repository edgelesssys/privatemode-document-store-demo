#!/usr/bin/env bash
set -euo pipefail

# Delete a full (unchunked) stored document via /full_documents/{collection}/{id}.
# Usage:
#   BASE_URL=http://localhost:8081 ./tests/smoke/full_documents/delete.sh

BASE_URL=${BASE_URL:-http://localhost:8081}
COLLECTION=${COLLECTION:-test-full-docs}
DOC_ID=${DOC_ID:-fulldoc-demo}

url="$BASE_URL/documents/$COLLECTION/$DOC_ID"
echo "[smoke] DELETE /documents/{collection}/{id} -> $url"
status=$(curl -sS -w '%{http_code}' -o /tmp/full_doc_delete_resp.json -X DELETE "$url")

# Response doesn't currently include body text
if [[ -s /tmp/full_doc_delete_resp.json ]]; then
  jq . /tmp/full_doc_delete_resp.json || cat /tmp/full_doc_delete_resp.json
fi
echo "Status: $status"

if [[ "$status" != "200" ]]; then
  echo "Delete full document failed" >&2
  exit 1
fi
