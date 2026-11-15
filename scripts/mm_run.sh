#!/usr/bin/env bash
# Keep `atx mm` running until a deadline. If the quoter exits early (crash, kill, error), RESTART it with --adopt so it
# takes over its own positions and keeps working them with passive quotes. Never market-flattens on a crash (user
# instruction 2026-11-08 23:07 UTC). The only market closes are the quoter's own, at the deadline / loss limit, after
# the passive wind-down window.
#
#   scripts/mm_run.sh 2026-11-09T21:00:00Z --config config/mm_multi.yaml --skip-busy [--yes]
set -u
cd "$(dirname "$0")/.."
UNTIL="$1"; shift
mkdir -p bin logs
cp -f target/release/atx.exe bin/atx-mm.exe 2>/dev/null || true
ENVN="${AI_TRADER_ENVIRONMENT:-$(grep -E '^environment:' config/settings.yaml | awk '{print $2}')}"
END=$(date -u -d "$UNTIL" +%s)
FIRST=1
# MM_ADOPT_FIRST=1: the first start already adopts existing positions (a planned swap of the quoter or its config)
if [ "${MM_ADOPT_FIRST:-0}" = "1" ]; then FIRST=0; fi
while [ "$(date -u +%s)" -lt "$END" ]; do
  LOG="logs/mm_${ENVN}_$(date -u +%Y%m%d).log"
  if [ $FIRST -eq 1 ]; then ADOPT=""; else ADOPT="--adopt"; fi
  echo "### mm start $(date -u +%FT%TZ) env=$ENVN until=$UNTIL args=$* $ADOPT" >> "$LOG"
  ./bin/atx-mm.exe mm --until "$UNTIL" "$@" $ADOPT >> "$LOG" 2>&1
  RC=$?
  echo "### mm exited $RC at $(date -u +%FT%TZ)" >> "$LOG"
  FIRST=0
  # a clean exit (deadline / loss limit / Ctrl-C) ends the run; anything else restarts in 5 s and adopts the book
  if [ $RC -eq 0 ]; then break; fi
  sleep 5
done
echo "### mm done $(date -u +%FT%TZ)" >> "logs/mm_${ENVN}_$(date -u +%Y%m%d).log"
