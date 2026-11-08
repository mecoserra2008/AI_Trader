# Always-on order-flow collector with auto-restart (Windows).
# Runs a private copy of atx so `cargo build` never fights a locked binary.
#
#   powershell -ExecutionPolicy Bypass -File scripts\collector.ps1
#
# Persistent setup (survives logoff/reboot): Task Scheduler -> Create Task ->
#   Trigger: At startup; Action: powershell.exe -ExecutionPolicy Bypass -File <repo>\scripts\collector.ps1
#   Settings: "If the task fails, restart every 1 minute". Add a second daily task for:
#   uv run python -m ai_trader.storage.flow maintain     (rollup + compaction + budget, e.g. 00:20 UTC)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force bin, logs | Out-Null
Copy-Item target\release\atx.exe bin\atx-collector.exe -Force
$env:RUST_LOG = "info"
while ($true) {
    $log = "logs\collector_$((Get-Date).ToUniversalTime().ToString('yyyyMMdd')).log"
    "### start $((Get-Date).ToUniversalTime().ToString('o'))" | Out-File -Append -Encoding utf8 $log
    & bin\atx-collector.exe collect *>> $log
    "### exited code $LASTEXITCODE; restarting in 10 s" | Out-File -Append -Encoding utf8 $log
    Start-Sleep -Seconds 10
}
