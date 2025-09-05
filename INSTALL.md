# Installation Guide

You can install Privatemode Document Store as a background service so it auto-starts at login and keeps running. This guide assumes you’re comfortable with terminals and git.

Requirements

- Python 3.12 or newer on your machine
- [Privatemode proxy](https://docs.privatemode.ai/guides/proxy-configuration) running on <http://localhost:8080> (default) or other host/port as configured. Make sure to enable [prompt caching](https://docs.privatemode.ai/guides/proxy-configuration#prompt-caching) in the proxy to reduce latency and cost.

Service environment defaults (override by setting env before install):

- HOST=127.0.0.1
- PORT=8081
- LOG_LEVEL=info
- PRIVATEMODE_API_KEY=(default: none, using the value configured in the Privatemode proxy)
- PRIVATEMODE_API_BASE=`http://localhost:8080/v1`

## macOS: background service (launchd)

Installs to your home directory and creates a Launch Agent that restarts on failure.

1. Install

    If you didn't clone the repo but downloaded it, you may have to allow modification and execution:

    ```bash
    chmod +xw ./scripts/setup/macos/*
    xattr -d com.apple.quarantine ./scripts/setup/macos/*
    ```

    ```bash
    ./scripts/setup/macos/install.sh
    ```

    Paths

    - Binary wrapper: ~/.local/bin/privatemode-document-store
    - Logs: ~/Library/Logs/privatemode-document-store/(stdout|stderr).log
    - LaunchAgent: ~/Library/LaunchAgents/com.privatemode.document-store.plist

2. Status & logs

    ```bash
    launchctl list | grep com.privatemode.document-store
    tail -f ~/Library/Logs/privatemode-document-store/stdout.log \\
    ~/Library/Logs/privatemode-document-store/stderr.log
    ```

3. Control

    ```bash
    launchctl unload -w ~/Library/LaunchAgents/com.privatemode.document-store.plist
    launchctl load -w ~/Library/LaunchAgents/com.privatemode.document-store.plist
    ```

4. Uninstall

    ```bash
    ./scripts/setup/macos/uninstall.sh
    ```

## Windows: background service (Scheduled Task)

### Prerequisites

- Windows 10 or later
- Python 3.12+ (add to PATH)

### Enabling PowerShell Script Execution

By default, Windows may block PowerShell scripts. To enable script execution:

1. Open PowerShell as Administrator (right-click, "Run as administrator").
2. Run:

   ```powershell
   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. When prompted, type `Y` and press Enter.
4. This sets the policy so **local scripts can run** while **downloaded scripts must be signed**. If you cloned this repo or created the script locally, you can unblock the script and run it:

    ```powershell
    Unblock-File -Path scripts/setup/windows/install.ps1
    scripts/setup/windows/install.ps1
    ```

    If you cannot change the policy or unblocking is not allowed, you can run the installer with:

    ```powershell
    powershell -ExecutionPolicy Bypass -File scripts/setup/windows/install.ps1
    ```

### Install

1. Open PowerShell. Make sure you have either unblocked the script (see above) or run with `-ExecutionPolicy Bypass`.
2. Run:

   ```powershell
   scripts/setup/windows/install.ps1
   ```

   This will:
   - Create a virtual environment in `%LOCALAPPDATA%\privatemode-document-store\venv`
   - Install dependencies
   - Register a Windows Scheduled Task to run the service in the background
   - Create a `run-document-store.cmd` wrapper for manual foreground runs

Paths

- App dir: %LOCALAPPDATA%\privatemode-document-store
- Logs:    %LOCALAPPDATA%\privatemode-document-store\logs\(stdout|stderr).log
- Data:    %LOCALAPPDATA%\privatemode-document-store\data\chroma

### Status & logs

```powershell
Get-ScheduledTask -TaskName PrivatemodeDocumentStore | Format-List *
Get-Content -Path "$env:LOCALAPPDATA/privatemode-document-store/logs/stdout.log" -Wait
Get-Content -Path "$env:LOCALAPPDATA/privatemode-document-store/logs/stderr.log" -Wait
```

### Control

```powershell
Start-ScheduledTask -TaskName PrivatemodeDocumentStore   # start
Stop-ScheduledTask  -TaskName PrivatemodeDocumentStore   # stop
Unregister-ScheduledTask -TaskName PrivatemodeDocumentStore -Confirm:$false  # uninstall task only
```

### Uninstall completely

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup/windows/uninstall.ps1
```

### Manual Foreground Run (Fallback)

If you can't use PowerShell, you can run the app manually (after install):

```cmd
@echo off
setlocal
set APP_DIR=%LOCALAPPDATA%\privatemode-document-store
set VENV_DIR=%APP_DIR%\venv
set PYEXE=%VENV_DIR%\Scripts\python.exe
if not exist "%PYEXE%" (
  echo Not installed. Run install.ps1 first.
  exit /b 1
)
"%PYEXE%" -m privatemode.document_store
```

Save as `run-document-store.cmd` and double-click to launch.

## Notes

- Both installers create and manage their own virtual environment; they don’t require admin privileges.
- To change ports or host later, edit the generated wrapper (macOS: ~/.local/bin/privatemode-document-store; Windows: %LOCALAPPDATA%\privatemode-document-store\run-document-store.cmd) and reload/restart the service.
- Health endpoint: <http://HOST:PORT/health>
- Telemetry: consider disabling any upstream telemetry (e.g., chromadb) if desired.
