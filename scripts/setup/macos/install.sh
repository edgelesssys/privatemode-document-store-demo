#!/usr/bin/env bash
set -euo pipefail

APP_ID="com.privatemode.document-store"
APP_NAME="privatemode-document-store"

PREFIX="${HOME}/.local"
APP_DIR="${HOME}/.local/share/${APP_NAME}"
VENV_DIR="${APP_DIR}/venv"
BIN_DIR="${HOME}/.local/bin"
PLIST="${HOME}/Library/LaunchAgents/${APP_ID}.plist"
LOG_DIR="${HOME}/Library/Logs/${APP_NAME}"

PYTHON_BIN="${PYTHON_BIN:-}"
PORT="${PORT:-8081}"
HOST="${HOST:-127.0.0.1}"
RELOAD="${RELOAD:-0}" # 1 to enable dev reload
PRIVATEMODE_API_KEY="${PRIVATEMODE_API_KEY:-}"
PRIVATEMODE_API_BASE="${PRIVATEMODE_API_BASE:-}"

mkdir -p "${APP_DIR}" "${BIN_DIR}" "${LOG_DIR}"

# Pick a suitable Python (3.12+) if not provided
if [[ -z "${PYTHON_BIN}" ]]; then
  # Prefer Homebrew Python
  if [[ -x "/opt/homebrew/bin/python3" ]]; then
    PYTHON_BIN="/opt/homebrew/bin/python3"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "ERROR: python3 not found. Install with: brew install python" >&2
    exit 1
  fi
fi

# Verify Python version >= 3.12
PY_VER_STR="$(${PYTHON_BIN} -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0)"
PY_VER_MAJOR="${PY_VER_STR%%.*}"
PY_VER_MINOR="${PY_VER_STR##*.}"
if [[ "${PY_VER_MAJOR}" -lt 3 || ( "${PY_VER_MAJOR}" -eq 3 && "${PY_VER_MINOR}" -lt 12 ) ]]; then
  echo "ERROR: Python ${PY_VER_STR} found at ${PYTHON_BIN}; need >= 3.12. Try: brew install python" >&2
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
LOG_DIR="${HOME}/Library/Logs/${APP_NAME}"
mkdir -p "${LOG_DIR}"
# Env: set defaults if not provided
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8081}"
export RELOAD="${RELOAD:-0}"
# Pass through any CLI args (e.g., --debug, --host, --port)
exec "${VENV_DIR}/bin/python" -m privatemode.document_store "$@"
SH
chmod +x "${BIN_DIR}/${APP_NAME}"

echo "[4/4] Installing launch agent at ${PLIST} and starting it"

# forward debug if in arg list
if [ "$#" -gt 0 ] && [[ "$*" == *"--debug"* ]]; then
    DEBUG_PARAM="<string>--debug</string>"
else
    DEBUG_PARAM=""
fi

# Build optional env vars for plist
OPTIONAL_ENV=""
if [[ -n "${PRIVATEMODE_API_KEY}" ]]; then
  OPTIONAL_ENV="${OPTIONAL_ENV}
    <key>PRIVATEMODE_API_KEY</key><string>${PRIVATEMODE_API_KEY}</string>"
fi
if [[ -n "${PRIVATEMODE_API_BASE}" ]]; then
  OPTIONAL_ENV="${OPTIONAL_ENV}
    <key>PRIVATEMODE_API_BASE</key><string>${PRIVATEMODE_API_BASE}</string>"
fi

# Render plist with current paths and defaults
cat > "${PLIST}" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${APP_ID}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${VENV_DIR}/bin/python</string>
    <string>-m</string>
  <string>privatemode.document_store</string>
    ${DEBUG_PARAM}
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOST</key><string>${HOST}</string>
    <key>PORT</key><string>${PORT}</string>
    <key>RELOAD</key><string>${RELOAD}</string>
${OPTIONAL_ENV}
  </dict>
  <key>WorkingDirectory</key><string>${APP_DIR}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>${LOG_DIR}/stdout.log</string>
  <key>StandardErrorPath</key><string>${LOG_DIR}/stderr.log</string>
</dict>
</plist>
PL

launchctl unload -w "${PLIST}" >/dev/null 2>&1 || true
launchctl load -w "${PLIST}"

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

USER_UID="$(id -u)"
status_line="$(launchctl list | awk -v label="${APP_ID}" '$3==label{print $0}')"
if [[ -n "${status_line}" ]]; then
  pid="$(awk '{print $1}' <<<"${status_line}")"
  if [[ "${pid}" == "-" ]]; then
    STATE_TEXT="${YELLOW}loaded (stopped)${RESET}"
  else
    STATE_TEXT="${GREEN}OK (pid ${pid})${RESET}"
  fi
else
  STATE_TEXT="${RED}not loaded${RESET}"
fi

printf "\n${BOLD}Installed${RESET} ${DIM}(service: ${APP_ID})${RESET}\n"
printf "  • Binary:   ${CYAN}%s${RESET}\n" "${BIN_DIR}/${APP_NAME}"
printf "  • Logs:     ${CYAN}%s${RESET}\n" "${LOG_DIR}/(stdout|stderr).log"
printf "  • Bind:     http://${HOST}:${PORT}\n"

printf "\n${BOLD}Status & logs${RESET}\n"
printf "  ${GRAY}# quick status${RESET}\n  launchctl list | grep %s\n" "${APP_ID}"
printf "  ${GRAY}# detailed state${RESET}\n  launchctl print gui/%s/%s | grep 'state ='\n" "${USER_UID}" "${APP_ID}"
printf "  ${GRAY}# follow logs${RESET}\n  tail -f %s %s\n" "${LOG_DIR}/stdout.log" "${LOG_DIR}/stderr.log"

printf "\n${BOLD}Control${RESET}\n"
printf "  ${GRAY}# restart now${RESET}\n  launchctl kickstart -k gui/%s/%s\n" "${USER_UID}" "${APP_ID}"
printf "  ${GRAY}# stop (unload)${RESET}\n  launchctl unload -w %s\n" "${PLIST}"
printf "  ${GRAY}# start (load)${RESET}\n  launchctl load -w %s\n" "${PLIST}"

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
  printf "\n${YELLOW}⚠ Smoke test had failures. Check logs:${RESET} ${LOG_DIR}/stderr.log ${LOG_DIR}/stdout.log\n"
fi
