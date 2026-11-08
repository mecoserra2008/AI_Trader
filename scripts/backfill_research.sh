#!/usr/bin/env bash
# Research backfill (v1, 2026-11-07). Idempotent enough for research: re-running writes duplicate rows
# with identical (pk, known_ts), which the as-of reader collapses. Compaction comes later.
#
# Survivorship caveat: this list is today's survivors. Results that compare assets cross-sectionally
# are biased toward coins that are still listed. Per-asset time-series results are much less affected.
set -u
cd "$(dirname "$0")/.."
X=./target/release/atx
LOG=logs/backfill_$(date -u +%Y%m%d_%H%M).log
mkdir -p logs
export RUST_LOG=warn

SYMS=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT,LTCUSDT,BCHUSDT,AVAXUSDT,DOTUSDT,BNBUSDT,TRXUSDT,NEARUSDT,ATOMUSDT,ETCUSDT,FILUSDT,UNIUSDT,AAVEUSDT,XLMUSDT,SUIUSDT,ENAUSDT,HYPEUSDT
START=2020-01-01

run() { echo "### $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

run $X instruments
for f in 1d 4h 1h; do run $X klines --symbols $SYMS --freq $f --start $START; done
run $X klines --symbols BTCUSDT,ETHUSDT --freq 15m --start 2024-01-01
run $X klines --symbols BTCUSDT,ETHUSDT --freq 1h --price-type index --start $START
run $X funding --symbols $SYMS --start $START
run $X oi --symbols $SYMS --interval 4h --start $START
run $X lsr --symbols $SYMS --period 4h --start $START
run $X alt fng
run $X alt stablecoins
run $X alt dvol --currencies BTC,ETH --resolution 3600 --start 2021-03-01
FROM=$(date -u -d '180 days ago' +%Y-%m-%d)
TO=$(date -u -d '1 day ago' +%Y-%m-%d)
for s in BTCUSDT ETHUSDT; do run $X trades --symbol $s --from $FROM --to $TO --freqs 1h,15m; done
echo "### DONE $(date -u)" | tee -a "$LOG"
