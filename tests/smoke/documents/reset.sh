#!/usr/bin/env bash
set -euo pipefail

# Reset the vector database via the admin endpoint and print the response.
# Usage:
#   BASE_URL=http://localhost:8081 ./tests/smoke/documents/reset.sh

BASE_URL=${BASE_URL:-http://localhost:8081}

echo "[smoke] POST /admin/reset -> $BASE_URL/admin/reset"
curl -sS -X POST "$BASE_URL/admin/reset" -H 'Content-Type: application/json'
echo
