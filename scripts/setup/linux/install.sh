#!/usr/bin/env bash
set -euo pipefail

APP_ID="com.privatemode.document-store"
APP_NAME="privatemode-document-store"

PREFIX="${HOME}/.local"
APP_DIR="${HOME}/.local/share/${APP_NAME}"
VENV_DIR="${APP_DIR}/venv"
BIN_DIR="${HOME}/.local/bin"
SERVICE_FILE="${HOME}/.config/systemd/user/${APP_ID}.service"
LOG_DIR="${HOME}/.local/share/${APP_NAME}/logs"

PYTHON_BIN="${PYTHON_BIN:-}"
PORT="${PORT:-8081}"
HOST="${HOST:-127.0.0.1}"
RELOAD="${RELOAD:-0}" # 1 to enable dev reload
PRIVATEMODE_API_KEY="${PRIVATEMODE_API_KEY:-}"
PRIVATEMODE_API_BASE="${PRIVATEMODE_API_BASE:-}"
VECTOR_DB_PATH="${APP_DIR}/data"

mkdir -p "${APP_DIR}" "${BIN_DIR}" "${LOG_DIR}"
chmod 755 "${APP_DIR}" "${BIN_DIR}" "${LOG_DIR}"

# Pick a suitable Python (3.12+) if not provided
if [[ -z "${PYTHON_BIN}" ]]; then
  # Prefer higher versions first, starting from 3.12
  for py in python3.12 python3; do
    if command -v "$py" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "$py")"
      break
    fi
  done
  if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: No suitable python3 found. Install with: apt install python3 python3-venv" >&2
    exit 1
  fi
fi

# Verify Python version >= 3.12
PY_VER_STR="$(${PYTHON_BIN} -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0)"
PY_VER_MAJOR="${PY_VER_STR%%.*}"
PY_VER_MINOR="${PY_VER_STR##*.}"
if [[ "${PY_VER_MAJOR}" -lt 3 || ( "${PY_VER_MAJOR}" -eq 3 && "${PY_VER_MINOR}" -lt 12 ) ]]; then
  echo "ERROR: Python ${PY_VER_STR} found at ${PYTHON_BIN}; need >= 3.12. Try: apt install python3.12 python3.12-venv" >&2
  exit 1
fi

echo "[1/4] Creating virtualenv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
"${VENV_DIR}/bin/python" -m pip install -U pip wheel

echo "[2/4] Installing this project into the venv"
# Install the package from the current repo checkout
"${VENV_DIR}/bin/python" -m pip install -e .

echo "[3/4] Creating CLI wrapper at ${BIN_DIR}/${APP_NAME}"
cat > "${BIN_DIR}/${APP_NAME}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
APP_NAME="privatemode-document-store"
APP_DIR="${HOME}/.local/share/${APP_NAME}"
VENV_DIR="${APP_DIR}/venv"
LOG_DIR="${HOME}/.local/share/${APP_NAME}/logs"
mkdir -p "${LOG_DIR}"
# Env: set defaults if not provided
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8081}"
export RELOAD="${RELOAD:-0}"
# Pass through any CLI args (e.g., --debug, --host, --port)
exec "${VENV_DIR}/bin/python" -m privatemode.document_store "$@"
SH
chmod +x "${BIN_DIR}/${APP_NAME}"

echo "[4/4] Installing systemd service at ${SERVICE_FILE} and starting it"

mkdir -p "${HOME}/.config/systemd/user"
# Set secure permissions on systemd user directory
chmod 755 "${HOME}/.config/systemd/user"

# forward debug if in arg list
if [ "$#" -gt 0 ] && [[ "$*" == *"--debug"* ]]; then
    DEBUG_PARAM="--debug"
else
    DEBUG_PARAM=""
fi

# Build optional env vars for service
OPTIONAL_ENV=""
if [[ -n "${PRIVATEMODE_API_KEY}" ]]; then
  OPTIONAL_ENV="${OPTIONAL_ENV}
Environment=PRIVATEMODE_API_KEY=${PRIVATEMODE_API_KEY}"
fi
if [[ -n "${PRIVATEMODE_API_BASE}" ]]; then
  OPTIONAL_ENV="${OPTIONAL_ENV}
Environment=PRIVATEMODE_API_BASE=${PRIVATEMODE_API_BASE}"
fi
# Ensure it ends with newline
OPTIONAL_ENV="${OPTIONAL_ENV}
"

# Render service file with current paths and defaults
cat > "${SERVICE_FILE}" <<SERVICE
[Unit]
Description=PrivateMode Document Store
After=network.target

[Service]
Type=simple
ExecStart=${VENV_DIR}/bin/python -m privatemode.document_store ${DEBUG_PARAM}
Environment=HOST=${HOST}
Environment=PORT=${PORT}
Environment=RELOAD=${RELOAD}
Environment=VECTOR_DB_PATH=${VECTOR_DB_PATH}
${OPTIONAL_ENV}WorkingDirectory=${APP_DIR}
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
SERVICE

# Set secure permissions on service file
chmod 644 "${SERVICE_FILE}"

systemctl --user daemon-reload
systemctl --user enable "${APP_ID}.service"
systemctl --user start "${APP_ID}.service"

# --- Nice status output ---

# Colors (only if stdout is a TTY)
if [[ -t 1 ]]; then
  BOLD="\033[1m"; DIM="\033[2m"; RESET="\033[0m"
  GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"; BLUE="\033[34m"; GRAY="\033[90m"
else
  BOLD=""; DIM=""; RESET=""; GREEN=""; YELLOW=""; RED=""; CYAN=""; BLUE=""; GRAY=""
fi

# Pretty, aligned status lines
LWIDTH=22
step_begin() {
  local label="$1"
  printf "%b %-*s %b%s%b" "${DIM}»${RESET}" "${LWIDTH}" "${label}..."
}

step_end() {
  local status="$1"; local detail="${2-}"
  # Print status, optional detail (prefixed with a space only if present), then reset and newline.
  printf "%b%b%b\n" "${status}" "${detail:+ ${detail}}" "${RESET}"
}

status_output="$(systemctl --user is-active "${APP_ID}.service" 2>/dev/null || echo failed)"
if [[ "${status_output}" == "active" ]]; then
  pid="$(systemctl --user show "${APP_ID}.service" -p MainPID --value)"
  STATE_TEXT="${GREEN}OK (pid ${pid})${RESET}"
elif [[ "${status_output}" == "failed" ]]; then
  STATE_TEXT="${RED}failed${RESET}"
else
  STATE_TEXT="${YELLOW}${status_output}${RESET}"
fi

printf "\n${BOLD}Installed${RESET} ${DIM}(service: ${APP_ID}.service)${RESET}\n"
printf "  • Binary:   ${CYAN}%s${RESET}\n" "${BIN_DIR}/${APP_NAME}"
printf "  • Logs:     ${CYAN}%s${RESET}\n" "journalctl --user -u ${APP_ID}.service"
printf "  • Bind:     http://${HOST}:${PORT}\n"

printf "\n${BOLD}Status & logs${RESET}\n"
printf "  ${GRAY}# quick status${RESET}\n  systemctl --user status %s\n" "${APP_ID}.service"
printf "  ${GRAY}# follow logs${RESET}\n  journalctl --user -u %s -f\n" "${APP_ID}.service"

printf "\n${BOLD}Control${RESET}\n"
printf "  ${GRAY}# restart now${RESET}\n  systemctl --user restart %s\n" "${APP_ID}.service"
printf "  ${GRAY}# stop${RESET}\n  systemctl --user stop %s\n" "${APP_ID}.service"
printf "  ${GRAY}# start${RESET}\n  systemctl --user start %s\n" "${APP_ID}.service"

printf "\n${BOLD}Smoke test${RESET}\n"
printf "  curl -sf http://%s:%s/health && echo\n\n" "${HOST}" "${PORT}"

# Run a simple end-to-end smoke test (non-fatal): health -> store -> query
BASE_URL="http://${HOST}:${PORT}"
SMOKE_OK=1

step_begin "Process"
step_end "${STATE_TEXT}"

# Give the server a moment, then poll /health for up to ~15s
step_begin "Health"
sleep 1
HEALTH_OK=0
for attempt in {1..15}; do
  if curl -sf "${BASE_URL}/health" >/dev/null; then
    HEALTH_OK=1
    break
  fi
  sleep 1
done
if [[ "${HEALTH_OK}" -eq 1 ]]; then
  step_end "${GREEN}OK${RESET}"
else
  step_end "${YELLOW}FAILED${RESET}"
  SMOKE_OK=0
fi

step_begin "Store test doc"
STORE_CODE=$(curl -sS -o /tmp/eds_install_store.json -w "%{http_code}" \
  -X POST "${BASE_URL}/documents" -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer test-key' \
  -d '{"collection":"docs","id":"install-smoke","text":"hello world","metadata":{}}') || STORE_CODE=000
if [[ "${STORE_CODE}" == "201" ]]; then
  step_end "${GREEN}OK${RESET}" "(201)"
else
  step_end "${YELLOW}FAILED${RESET}" "(${STORE_CODE})"
  SMOKE_OK=0
fi

step_begin "Query"
QUERY_CODE=$(curl -sS -o /tmp/eds_install_query.json -w "%{http_code}" \
  -X POST "${BASE_URL}/retrieval/query" -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer test-key' \
  -d '{"collection":"docs","query":"hello world","top_k":5}') || QUERY_CODE=000
if [[ "${QUERY_CODE}" == "200" ]]; then
  step_end "${GREEN}OK${RESET}" "(200)"
else
  step_end "${YELLOW}FAILED${RESET}" "(${QUERY_CODE})"
  SMOKE_OK=0
fi

if [[ "${SMOKE_OK}" -eq 1 ]]; then
  printf "\n${GREEN}✓ Tests passed.${RESET}\n"
else
  printf "\n${YELLOW}⚠ Smoke test had failures. Check logs:${RESET} journalctl --user -u ${APP_ID}.service\n"
fi
