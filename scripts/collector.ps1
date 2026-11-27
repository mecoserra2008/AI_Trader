# Always-on order-flow collector with auto-restart (Windows).
# Runs a private copy of atx so `cargo build` never fights a locked binary.
#
#   powershell -ExecutionPolicy Bypass -File scripts\collector.ps1
#
# No Task Scheduler / Startup-folder entry (standing user constraint): the desk normally starts the collector with
# `bash scripts/ops/desk_start.sh`, which must be run again after every reboot. The daily compaction + 1-minute rollup
# runs there as the `daily_ops` loop (python -m ai_trader.storage.daily).

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
