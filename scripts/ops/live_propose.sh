#!/usr/bin/env bash
# Risk-check a live intent (nothing is sent). Usage: bash scripts/ops/live_propose.sh <intent.json>
cd "C:/Users/Asus/Desktop/AI_Trader" && AI_TRADER_ENVIRONMENT=mainnet AI_TRADER_ALLOW_MAINNET=1 target/debug/atx.exe live propose --file "$1" 2>&1 | grep -v -iE "balance|equity|wallet|available"
