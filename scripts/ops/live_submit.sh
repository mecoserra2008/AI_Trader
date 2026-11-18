#!/usr/bin/env bash
# Submit a pending live proposal. atx itself refuses unless the user approved it (approvals/<id>.ok from `atx live approve`),
# a valid user-signed delegation covers it (`atx live delegate`), or the auto gate is open. Every risk limit is re-checked.
# Usage: bash scripts/ops/live_submit.sh <proposal_id>
cd "C:/Users/Asus/Desktop/AI_Trader" && AI_TRADER_ENVIRONMENT=mainnet AI_TRADER_ALLOW_MAINNET=1 timeout 1000 target/debug/atx.exe live submit --id "$1" 2>&1 | grep -v -iE "balance|equity|wallet|available"
