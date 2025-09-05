#!/usr/bin/env bash
set -euo pipefail

# Query the retrieval API and print the response.
# Usage:
#   BASE_URL=http://localhost:8081 ./tests/smoke/documents/retrieve.sh

BASE_URL=${BASE_URL:-http://localhost:8081}

echo "[smoke] POST /retrieval/query -> $BASE_URL/retrieval/query"
response=$(curl -sS -w "%{http_code}" -o /tmp/resp.json -X POST "$BASE_URL/retrieval/query" \
  -H 'Content-Type: application/json' \
  -d '{"collection": "test-docs", "query":"How does backpressure work in Kafka?","top_k":50}')
jq . /tmp/resp.json
echo "Status: $response"
echo
