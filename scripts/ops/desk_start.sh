#!/usr/bin/env bash
# Start every desk process detached (survives session interrupts). PIDs -> logs/run/<name>.pid. Stop: scripts/ops/desk_stop.sh. Usage: desk_start.sh [name ...]
# No OS auto-start (no Task Scheduler / Startup-folder entry, a standing user constraint): run this script again after every reboot.
cd "C:/Users/Asus/Desktop/AI_Trader"
ONLY=" $* "   # optional: names to start (e.g. `desk_start.sh daily_ops`); none = all
start() { name=$1; shift; [ "$ONLY" != "  " ] && [[ "$ONLY" != *" $name "* ]] && return 0; nohup bash -c "$*" >/dev/null 2>&1 & echo $! > logs/run/$name.pid; echo "$name msys pid $!"; }
export RUST_LOG=info PYTHONIOENCODING=utf-8
start updater   'while true; do L=logs/updater_$(date -u +%Y%m%d).log; echo "### start $(date -u +%FT%TZ)" >> $L; ./bin/atx-updater.exe update --loop >> $L 2>&1; echo "### exited $?; restart in 10s" >> $L; sleep 10; done'
start collector 'while true; do L=logs/collector_$(date -u +%Y%m%d).log; echo "### start $(date -u +%FT%TZ)" >> $L; ./bin/atx-collector.exe collect >> $L 2>&1; echo "### exited $?; restart in 10s" >> $L; sleep 10; done'
start live_nn   'while true; do now=$(date -u +%s); next=$(( (now/3600+1)*3600 + 180 )); while [ $(date -u +%s) -lt $next ]; do sleep 30; done; L=logs/live_nn_$(date -u +%Y%m%d).log; want="nn-$(date -u +%Y%m%dT%H)"; echo "### $(date -u +%FT%TZ)" >> $L; uv run --extra ml python scripts/live_nn.py >> $L 2>&1; echo "### exit $?" >> $L; if ! head -1 research/live/LATEST.md | grep -q "$want"; then sleep 300; echo "### $(date -u +%FT%TZ) retry (wanted $want)" >> $L; uv run --extra ml python scripts/live_nn.py >> $L 2>&1; echo "### exit $?" >> $L; fi; done'
start vol_ml    'while true; do now=$(date -u +%s); next=$(( (now/3600+1)*3600 + 60 )); while [ $(date -u +%s) -lt $next ]; do sleep 30; done; echo "### $(date -u +%FT%TZ)" >> logs/vol_ml_forward.log; uv run --extra ml python scripts/vol_ml_log.py >> logs/vol_ml_forward.log 2>&1; done'
start news_alt  'while true; do now=$(date -u +%s); next=$(( (now/86400+1)*86400 + 4500 )); while [ $(date -u +%s) -lt $next ]; do sleep 30; done; echo "### $(date -u +%FT%TZ)" >> logs/news_alt_update.log; uv run python -c "from ai_trader.news.alt import update; print(update())" >> logs/news_alt_update.log 2>&1; done'
start desk_collect 'uv run python scripts/desk_collect.py --rss-counts >> logs/desk_collect.log 2>&1'
start level_watch  'uv run python scripts/level_watch.py --levels 67376.52 65904.12 65270.68 64494.49 63836.28 63000 62067.96 61899.2 57756 57719.9 56799.9 56000 55000 >> logs/level_watch.stdout.log 2>&1'
start dashboard    'uv run --extra ml python scripts/live_dashboard.py >> logs/live_dashboard.log 2>&1'
start tape         './target/scalp/atx-tape.exe tape >> logs/tape.stdout.log 2>&1'
start xvenue       '.venv/Scripts/python.exe scripts/xvenue_tape.py --symbols KAVA SOPH RARE >> logs/xvenue_tape.stdout.log 2>&1'
start xs_v4        'uv run --extra ml python scripts/live_xs_v4.py --loop >> data/journal/live_xs_v4/loop.log 2>&1'
# Plan time stops (intent context.time_stop_h): closes due BOT trades reduce-only, never opens anything. Runs from a copy
# (bin/atx-live.exe) so builds don't hit a locked binary; refresh the copy after changing crates/atx/src/live.rs.
start watchdog  'while true; do L=logs/watchdog_$(date -u +%Y%m%d).log; echo "### start $(date -u +%FT%TZ)" >> $L; AI_TRADER_ENVIRONMENT=mainnet AI_TRADER_ALLOW_MAINNET=1 ./bin/atx-live.exe live watchdog --interval 60 2>&1 | grep --line-buffered -v -iE "balance|equity|wallet|available" >> $L; echo "### exited; restart in 30s" >> $L; sleep 30; done'
# D8 daily data jobs (ai_trader.storage.daily): cold flow_3s days compacted + rolled up to flow_1m (reversible), the last
# 2 days of flow_bars from the trade archive, the 08:00 Deribit surface (deribit_svi). 08:20 UTC daily, wall-clock
# polling every 60 s (suspend-safe), through the governor; atx runs from bin/atx-ops.exe (a copy, like the others).
start daily_ops 'while true; do now=$(date -u +%s); next=$(( now/86400*86400 + 30000 )); if [ $now -ge $next ]; then next=$(( next + 86400 )); fi; while [ $(date -u +%s) -lt $next ]; do sleep 60; done; echo "### $(date -u +%FT%TZ)" >> logs/daily_ops.log; PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/ops/governor.py run --mb 2500 --slots heavy --name daily_ops -- .venv/Scripts/python.exe -m ai_trader.storage.daily >> logs/daily_ops.log 2>&1; echo "### exit $?" >> logs/daily_ops.log; done'
# D1 stop-sigma shadow (ai_trader.risk.sigma): hourly at hh:02:30, wall-clock polling every 30 s (suspend-safe); appends
# data/journal/sigma_shadow/<date>.jsonl, refreshes the 5m RV store, then scores and takes the one-time switch gate.
start sigma_shadow 'uv run python scripts/sigma_shadow.py --loop >> logs/sigma_shadow.log 2>&1'
# D2 / D3 live entry gates (src/ai_trader/live/gates.py): data/live/gates.json every 60 s on the wall clock (suspend-safe).
# atx live refuses every new entry while the file is missing or older than 5 min, so start this BEFORE deploying an atx
# binary that reads it; closes, stops, reconcile and the watchdog never read it.
start live_gates   '.venv/Scripts/python.exe scripts/live_gates.py --loop >> logs/live_gates.log 2>&1'
