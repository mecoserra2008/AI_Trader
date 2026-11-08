#!/usr/bin/env bash
# Run `atx mm` until a deadline, then ALWAYS flatten the configured symbols, whatever the exit reason
# (deadline, loss limit, error, crash, kill). Environment comes from settings.yaml / AI_TRADER_ENVIRONMENT.
#
#   scripts/mm_run.sh 2026-11-08T12:00:00Z            # extra args are passed to atx mm (e.g. --yes)
set -u
cd "$(dirname "$0")/.."
UNTIL="$1"; shift
mkdir -p bin logs
cp -f target/release/atx.exe bin/atx-mm.exe
ENVN="${AI_TRADER_ENVIRONMENT:-$(grep -E '^environment:' config/settings.yaml | awk '{print $2}')}"
LOG="logs/mm_${ENVN}_$(date -u +%Y%m%d).log"
echo "### mm start $(date -u +%FT%TZ) env=$ENVN until=$UNTIL" >> "$LOG"
./bin/atx-mm.exe mm --until "$UNTIL" "$@" >> "$LOG" 2>&1
echo "### mm exited $? at $(date -u +%FT%TZ); safety flatten" >> "$LOG"
SYMS=$(grep -E '^symbols:' config/mm.yaml | sed -E 's/^symbols: *\[([^]]*)\].*/\1/' | tr -d ' ' | tr ',' ' ')
for s in $SYMS; do
  for i in 1 2 3; do
    ./bin/atx-mm.exe flatten --symbol "$s" --yes --session-id mm-safety >> "$LOG" 2>&1 && break
    sleep 5
  done
done
echo "### mm done $(date -u +%FT%TZ)" >> "$LOG"
