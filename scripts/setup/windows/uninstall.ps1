#Requires -Version 5.1
<#
Uninstall the Privatemode Document Store Windows background service.

Removes the Scheduled Task and deletes the local app directory under %LOCALAPPDATA%\privatemode-document-store.
Usage:
  powershell -ExecutionPolicy Bypass -File scripts/setup/windows/uninstall.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$AppName = 'privatemode-document-store'
$TaskName = 'PrivatemodeDocumentStore'
$AppDir = Join-Path $env:LOCALAPPDATA $AppName

try {
  if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue | Out-Null
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false | Out-Null
    Write-Host "Removed Scheduled Task: $TaskName"
  } else {
    Write-Host "Scheduled Task not found: $TaskName"
  }
} catch {
  Write-Host "Warning: failed to remove scheduled task: $_" -ForegroundColor Yellow
}

if (Test-Path $AppDir) {
  try {
    Remove-Item -Recurse -Force -Path $AppDir
    Write-Host "Removed app directory: $AppDir"
  } catch {
    Write-Host "Warning: failed to remove app directory. Delete manually: $AppDir" -ForegroundColor Yellow
  }
}

Write-Host "Uninstall complete."
