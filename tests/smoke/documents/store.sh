#!/usr/bin/env bash
set -euo pipefail

# Store a document via the API and print the response.
# Usage:
#   BASE_URL=http://localhost:8081 ./tests/smoke/documents/store.sh

BASE_URL=${BASE_URL:-http://localhost:8081}

echo "[smoke] POST /documents -> $BASE_URL/documents"
response=$(curl -sS -w "%{http_code}" -o /tmp/store_resp.json -X POST "$BASE_URL/documents" \
  -H 'Content-Type: application/json' \
  -d @- <<'JSON'
{
  "collection": "test-docs",
  "id": "doc-kafka",
  "text": "Kafka uses backpressure by controlling fetch rates and consumer lag. Kafka consumers apply backpressure when lag increases. Kafka uses backpressure by controlling fetch rates and consumer lag.",
  "metadata": {"title": "Operating Kafka at Scale", "namespace": "docs"}
}
JSON
)
jq . /tmp/store_resp.json
echo "Status: $response"
echo
