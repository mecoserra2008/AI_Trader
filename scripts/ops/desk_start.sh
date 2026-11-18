#!/usr/bin/env bash
# Start every desk process detached (survives session interrupts). PIDs -> logs/run/<name>.pid. Stop: scripts/ops/desk_stop.sh
cd "C:/Users/Asus/Desktop/AI_Trader"
start() { name=$1; shift; nohup bash -c "$*" >/dev/null 2>&1 & echo $! > logs/run/$name.pid; echo "$name msys pid $!"; }
export RUST_LOG=info PYTHONIOENCODING=utf-8
start updater   'while true; do L=logs/updater_$(date -u +%Y%m%d).log; echo "### start $(date -u +%FT%TZ)" >> $L; ./bin/atx-updater.exe update --loop >> $L 2>&1; echo "### exited $?; restart in 10s" >> $L; sleep 10; done'
start collector 'while true; do L=logs/collector_$(date -u +%Y%m%d).log; echo "### start $(date -u +%FT%TZ)" >> $L; ./bin/atx-collector.exe collect >> $L 2>&1; echo "### exited $?; restart in 10s" >> $L; sleep 10; done'
start live_nn   'while true; do now=$(date -u +%s); next=$(( (now/3600+1)*3600 + 180 )); sleep $(( next - now )); L=logs/live_nn_$(date -u +%Y%m%d).log; want="nn-$(date -u +%Y%m%dT%H)"; echo "### $(date -u +%FT%TZ)" >> $L; uv run --extra ml python scripts/live_nn.py >> $L 2>&1; echo "### exit $?" >> $L; if ! head -1 research/live/LATEST.md | grep -q "$want"; then sleep 300; echo "### $(date -u +%FT%TZ) retry (wanted $want)" >> $L; uv run --extra ml python scripts/live_nn.py >> $L 2>&1; echo "### exit $?" >> $L; fi; done'
start vol_ml    'while true; do now=$(date -u +%s); next=$(( (now/3600+1)*3600 + 60 )); sleep $(( next - now )); echo "### $(date -u +%FT%TZ)" >> logs/vol_ml_forward.log; uv run --extra ml python scripts/vol_ml_log.py >> logs/vol_ml_forward.log 2>&1; done'
start news_alt  'while true; do now=$(date -u +%s); next=$(( (now/86400+1)*86400 + 4500 )); sleep $(( next - now )); echo "### $(date -u +%FT%TZ)" >> logs/news_alt_update.log; uv run python -c "from ai_trader.news.alt import update; print(update())" >> logs/news_alt_update.log 2>&1; done'
start desk_collect 'uv run python scripts/desk_collect.py --rss-counts >> logs/desk_collect.log 2>&1'
start level_watch  'uv run python scripts/level_watch.py --levels 67376.52 65904.12 65270.68 64494.49 63836.28 63000 62067.96 61899.2 57756 >> logs/level_watch.stdout.log 2>&1'
start dashboard    'uv run --extra ml python scripts/live_dashboard.py >> logs/live_dashboard.log 2>&1'
start tape         './target/scalp/atx-tape.exe tape >> logs/tape.stdout.log 2>&1'
start xvenue       '.venv/Scripts/python.exe scripts/xvenue_tape.py --symbols KAVA SOPH RARE >> logs/xvenue_tape.stdout.log 2>&1'
start xs_v4        'uv run --extra ml python scripts/live_xs_v4.py --loop >> data/journal/live_xs_v4/loop.log 2>&1'
