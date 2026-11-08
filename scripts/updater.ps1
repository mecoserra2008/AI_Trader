# Always-on live market-data updater with auto-restart (Windows): bars 1h/4h/1d (last+mark),
# funding, open interest, Deribit DVOL, Fear & Greed -> lake. Config: settings.yaml `live:`.
# Runs a private copy of atx so `cargo build` never fights a locked binary.
#
#   powershell -ExecutionPolicy Bypass -File scripts\updater.ps1
#
# Persistent setup (survives logoff/reboot): Task Scheduler -> Create Task ->
#   Trigger: At startup; Action: powershell.exe -ExecutionPolicy Bypass -File <repo>\scripts\updater.ps1
#   Settings: "If the task fails, restart every 1 minute".
# Check staleness any time: PYTHONIOENCODING=utf-8 uv run python scripts/freshness.py

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force bin, logs | Out-Null
# Newest build wins (target\updater is the fallback target dir used when target\release is locked).
$src = Get-ChildItem target\release\atx.exe, target\updater\release\atx.exe -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($src) { Copy-Item $src.FullName bin\atx-updater.exe -Force }
if (-not (Test-Path bin\atx-updater.exe)) { throw "no atx.exe build found: run cargo build --release" }
$env:RUST_LOG = "info"
while ($true) {
    $log = "logs\updater_$((Get-Date).ToUniversalTime().ToString('yyyyMMdd')).log"
    "### start $((Get-Date).ToUniversalTime().ToString('o'))" | Out-File -Append -Encoding utf8 $log
    & bin\atx-updater.exe update --loop *>> $log
    "### exited code $LASTEXITCODE; restarting in 10 s" | Out-File -Append -Encoding utf8 $log
    Start-Sleep -Seconds 10
}
