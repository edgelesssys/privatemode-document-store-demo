#!/usr/bin/env bash
set -euo pipefail

# Store a document via the API and print the response.
# Usage:
#   BASE_URL=http://localhost:8081 ./tests/e2e/store_large_doc.sh

BASE_URL=${BASE_URL:-http://localhost:8081}

# Load text from large_doc.md (in the same directory as this script)
SCRIPT_DIR=$(dirname "$0")
DOC_FILE="$SCRIPT_DIR/large_doc.md"
if [[ ! -f "$DOC_FILE" ]]; then
  echo "Error: $DOC_FILE not found" >&2
  exit 1
fi

TEXT=$(cat "$DOC_FILE")

# Build JSON payload
PAYLOAD=$(jq -n --arg text "$TEXT" '{
  collection: "docs",
  id: "doc-large",
  text: $text,
  metadata: {title: "Large Doc", namespace: "docs"}
}')

echo "[e2e] POST /documents -> $BASE_URL/documents"
response=$(curl -sS -w "%{http_code}" -o /tmp/store_resp.json -X POST "$BASE_URL/documents" \
  -H 'Content-Type: application/json' \
  -d "$PAYLOAD")
jq . /tmp/store_resp.json
echo "Status: $response"
echo
