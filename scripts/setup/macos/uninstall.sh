#!/usr/bin/env bash
set -euo pipefail

APP_ID="com.privatemode.document-store"
APP_NAME="privatemode-document-store"
PLIST="${HOME}/Library/LaunchAgents/${APP_ID}.plist"
BIN="${HOME}/.local/bin/${APP_NAME}"
APP_DIR="${HOME}/.local/share/${APP_NAME}"
LOG_DIR="${HOME}/Library/Logs/${APP_NAME}"

# Colors for nicer output
if [[ -t 1 ]]; then
	BOLD="\033[1m"; RESET="\033[0m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"; GRAY="\033[90m"
else
	BOLD=""; RESET=""; GREEN=""; YELLOW=""; RED=""; CYAN=""; GRAY=""
fi

echo -e "${BOLD}Unloading launch agent...${RESET}"
UID_NUM="$(id -u)"
if [[ -f "${PLIST}" ]]; then
	launchctl bootout gui/${UID_NUM} "${PLIST}" >/dev/null 2>&1 || \
		launchctl unload -w "${PLIST}" >/dev/null 2>&1 || true
fi
# Also attempt removal by label (idempotent)
if launchctl list | awk '{print $3}' | grep -Fxq "${APP_ID}"; then
	launchctl remove "${APP_ID}" >/dev/null 2>&1 || true
fi
if launchctl list | awk '{print $3}' | grep -Fxq "${APP_ID}"; then
	echo -e "${YELLOW}Warning: launch agent label still present after removal attempts.${RESET}"
fi

echo -e "${BOLD}Checking for lingering processes...${RESET}"
# Identify PIDs whose command points to this app's venv running privatemode.document_store.
# Use subshell with 'set +e' to avoid aborting when grep finds no matches.
PIDS="$(ps -ax -o pid= -o command= 2>/dev/null | grep -F "${APP_DIR}/venv/bin/python" | grep -F privatemode.document_store | awk '{print $1}' | tr '\n' ' ' || true)"
if [[ -z "${PIDS// }" ]]; then
	# Fallback: broader match, but still constrained to module name
	PIDS="$(pgrep -f 'privatemode.document_store' || true)"
fi

if [[ -n "${PIDS// }" ]]; then
	echo -e "Found running PIDs: ${PIDS}. Sending SIGTERM..."
	for p in ${PIDS}; do
		kill "$p" 2>/dev/null || true
	done
	# Wait up to 5s for graceful exit
	for i in 1 2 3 4 5; do
		sleep 1
		STILL="$(for pid in ${PIDS}; do kill -0 "$pid" 2>/dev/null && echo -n "$pid "; done)"
		if [[ -z "${STILL// }" ]]; then
			break
		fi
	done
	# Force kill if still around
	STILL="$(for pid in ${PIDS}; do kill -0 "$pid" 2>/dev/null && echo -n "$pid "; done)"
	if [[ -n "${STILL// }" ]]; then
		echo -e "${YELLOW}Forcibly killing PIDs: ${STILL}${RESET}"
		for p in ${STILL}; do
			kill -9 "$p" 2>/dev/null || true
		done
	else
		echo -e "${GREEN}All processes exited cleanly.${RESET}"
	fi
else
	echo -e "${GRAY}No lingering processes found.${RESET}"
fi
echo -e "${BOLD}Removing files...${RESET}"
rm -f "${PLIST}" "${BIN}"
rm -rf "${APP_DIR}"

# Detect stray plists elsewhere that could respawn the service
OTHER_PLISTS=()
for d in "$HOME/Library/LaunchAgents" /Library/LaunchAgents /Library/LaunchDaemons; do
	[[ -d "$d" ]] || continue
	while IFS= read -r f; do
		[[ -z "$f" ]] && continue
		if grep -q "${APP_ID}" "$f" 2>/dev/null; then
			OTHER_PLISTS+=("$f")
		fi
	done < <(ls -1 "$d"/*.plist 2>/dev/null || true)
done
if [[ ${#OTHER_PLISTS[@]} -gt 0 ]]; then
	echo -e "${YELLOW}Potential additional plist files referencing ${APP_ID}:${RESET}"
	for f in "${OTHER_PLISTS[@]}"; do
		echo "  $f"
	done
	echo "You may need to remove these (some may require sudo)."
fi

# Ask whether to delete logs (default Yes). In non-interactive shells, delete automatically.
if [[ -d "${LOG_DIR}" ]]; then
	echo
	if [[ -t 0 ]]; then
		read -r -p "Delete logs in ${LOG_DIR}? [Y/n]: " response
		response=${response:-Y}
			case "${response}" in
				[Yy]*)
					rm -rf "${LOG_DIR}" && echo -e "${GREEN}Logs deleted.${RESET}" ;;
				*)
					echo -e "${YELLOW}Keeping logs in ${LOG_DIR}.${RESET}" ;;
			esac
	else
		echo -e "${GRAY}Non-interactive shell detected. Deleting logs in ${LOG_DIR}.${RESET}"
		rm -rf "${LOG_DIR}" && echo -e "${GREEN}Logs deleted.${RESET}"
	fi
fi

echo -e "\n${GREEN}Uninstall complete.${RESET}"

echo -e "\n${BOLD}Post-uninstall status check:${RESET}"
if lsof -nP -iTCP:8081 -sTCP:LISTEN 2>/dev/null | grep -q ':'; then
	echo -e "${YELLOW}Port 8081 is still in use. Another service may be running.${RESET}"
	lsof -nP -iTCP:8081 -sTCP:LISTEN 2>/dev/null | sed 's/^/  /'
else
	echo -e "${GREEN}Port 8081 is free.${RESET}"
fi
