#!/usr/bin/env bash
set -euo pipefail

# Run full document smoke sequence: store -> get -> delete -> get (expect 404)
# Usage:
#   BASE_URL=http://localhost:8081 ./tests/smoke/full_documents/run_all.sh

BASE_URL=${BASE_URL:-http://localhost:8081}
COLLECTION=${COLLECTION:-docs}
DOC_ID=${DOC_ID:-fulldoc-demo-$(date +%s)}
export BASE_URL COLLECTION DOC_ID

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$SCRIPT_DIR/store.sh"
"$SCRIPT_DIR/get.sh"
"$SCRIPT_DIR/delete.sh"

# Final fetch should 404
url="$BASE_URL/full_documents/$COLLECTION/$DOC_ID"
echo "[smoke] GET (expect 404) $url"
status=$(curl -sS -w '%{http_code}' -o /tmp/full_doc_final_get_resp.json "$url" || true)

if [[ "$status" == "404" ]]; then
  echo "Final GET correctly returned 404"
else
  echo "Expected 404, got $status" >&2
  exit 1
fi
