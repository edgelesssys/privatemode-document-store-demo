#Requires -Version 5.1
<#
Privatemode Document Store - Windows install script

What this does:
- Finds a Python 3.12+ interpreter (or uses $env:PYTHON_BIN if set)
- Creates a virtualenv under %LOCALAPPDATA%\privatemode-document-store\venv
- Installs this project into the venv (editable)
- Creates a run wrapper (run-document-store.cmd) that sets HOST/PORT/RELOAD and logs to %LOCALAPPDATA%\privatemode-document-store\logs
- Registers a per-user Scheduled Task that starts at logon and restarts on failure

Environment variables honored at install time (baked into the wrapper):
- HOST (default 127.0.0.1)
- PORT (default 8081)
- RELOAD (default 0)
- LOG_LEVEL (default info)

Usage:
  powershell -ExecutionPolicy Bypass -File scripts/setup/windows/install.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$AppName = 'privatemode-document-store'
$TaskName = 'PrivatemodeDocumentStore'
$AppId = 'com.privatemode.document-store'

$AppDir = Join-Path $env:LOCALAPPDATA $AppName
$VenvDir = Join-Path $AppDir 'venv'
$LogDir = Join-Path $AppDir 'logs'
$RunCmd = Join-Path $AppDir 'run-document-store.cmd'

function Test-Health([string]$Url, [int]$Tries=15) {
  for ($i=0; $i -lt $Tries; $i++) {
    try {
      $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
      if ($r.StatusCode -eq 200) { return $true }
    } catch {}
    Start-Sleep -Milliseconds 800
  }
  return $false
}

New-Item -ItemType Directory -Path $AppDir,$LogDir -Force | Out-Null

function Get-Python312 {
  param([string]$Preferred)
  if ($Preferred) {
    try {
      $ver = & $Preferred -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>$null
      if ($LASTEXITCODE -eq 0 -and $ver -ge '3.12') { return $Preferred }
    } catch {}
  }
  # Try the Python launcher
  if (Get-Command py -ErrorAction SilentlyContinue) {
    try {
      $exe = & py -3.12 -c "import sys;print(sys.executable)" 2>$null
      if ($LASTEXITCODE -eq 0 -and $exe) { return $exe }
    } catch {}
    try {
      $exe = & py -3 -c "import sys;import sys;print(sys.executable if sys.version_info>=(3,12) else '')" 2>$null
      if ($LASTEXITCODE -eq 0 -and $exe) { return $exe }
    } catch {}
  }
  foreach ($cmd in @('python3','python')) {
    $p = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($p) {
      try {
        $ver = & $p.Source -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver -ge '3.12') { return $p.Source }
      } catch {}
    }
  }
  throw 'Python 3.12+ not found. Install Python 3.12 from https://www.python.org/downloads/windows/ or Microsoft Store.'
}

$PreferredPython = $env:PYTHON_BIN
$Python = Get-Python312 -Preferred $PreferredPython
Write-Host "[1/5] Using Python: $Python"

Write-Host "[2/5] Creating virtualenv at $VenvDir"
& $Python -m venv "$VenvDir"
if ($LASTEXITCODE -ne 0) { throw "Failed to create virtualenv" }

$PyExe = Join-Path $VenvDir 'Scripts/python.exe'
$PipExe = Join-Path $VenvDir 'Scripts/pip.exe'

& $PyExe -m pip install -U pip wheel | Write-Host

Write-Host "[3/5] Installing this project into the venv (editable)"
# Ensure we install from the repository root (script is under scripts/setup/windows)
$RepoRoot = (Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent)
Push-Location $RepoRoot
try {
  & $PyExe -m pip install -e . | Write-Host
} finally {
  Pop-Location
}

# Prepare wrapper with configured env vars
$APP_HOST = if ($env:HOST) { $env:HOST } else { '127.0.0.1' }
$APP_PORT = if ($env:PORT) { $env:PORT } else { '8081' }
$APP_RELOAD = if ($env:RELOAD) { $env:RELOAD } else { '0' }
$APP_LOG_LEVEL = if ($env:LOG_LEVEL) { $env:LOG_LEVEL } else { 'info' }

Write-Host "[4/5] Creating run wrapper: $RunCmd"
$cmdContent = @"
@echo off
setlocal enableextensions
set APP_DIR=%LOCALAPPDATA%\%AppName%
set VENV_DIR=%APP_DIR%\venv
set LOG_DIR=%APP_DIR%\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

rem service environment (baked at install time; edit this file to change)
set HOST=$APP_HOST
set PORT=$APP_PORT
set RELOAD=$APP_RELOAD
set LOG_LEVEL=$APP_LOG_LEVEL
set VECTOR_DB_PATH=%APP_DIR%\data\chroma
if not exist "%VECTOR_DB_PATH%" mkdir "%VECTOR_DB_PATH%" >nul 2>&1

set PYEXE=%VENV_DIR%\Scripts\python.exe
if not exist "%PYEXE%" (
  echo Python not found in venv: %PYEXE%
  exit /b 1
)

rem Use start to detach a hidden window when launched by Task Scheduler
pushd "%APP_DIR%" >nul
"%PYEXE%" -m privatemode.document_store >> "%LOG_DIR%\stdout.log" 2>> "%LOG_DIR%\stderr.log"
popd >nul
"@

Set-Content -Path $RunCmd -Value $cmdContent -Encoding Ascii

# Create Scheduled Task (per-user, at logon, restart on failure)
Write-Host "[5/5] Registering Scheduled Task: $TaskName"
$action   = New-ScheduledTaskAction -Execute $RunCmd -WorkingDirectory $AppDir
$trigger  = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 999

try {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false | Out-Null
    }
} catch {}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Description "Privatemode Document Store background service" | Out-Null

# Start now and basic health probe
Start-ScheduledTask -TaskName $TaskName

Write-Host "\nInstalled (service: $TaskName)"
Write-Host "  • App dir:  $AppDir"
Write-Host "  • Logs:     $LogDir\(stdout|stderr).log"
Write-Host "  • Bind:     http://${APP_HOST}:${APP_PORT}"

Start-Sleep -Seconds 2

$base = "http://${APP_HOST}:${APP_PORT}"
if (Test-Health "${base}/health") {
    Write-Host "Health OK" -ForegroundColor Green
} else {
    Write-Host "Health check failed. Check logs:" -ForegroundColor Yellow
    Write-Host "  $LogDir\\stderr.log"
}

Write-Host "\nControl (PowerShell):"
Write-Host "  Get-ScheduledTask -TaskName $TaskName | Format-List *"
Write-Host "  Start-ScheduledTask -TaskName $TaskName  # start"
Write-Host "  Stop-ScheduledTask  -TaskName $TaskName  # stop"
Write-Host "  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false  # uninstall"
