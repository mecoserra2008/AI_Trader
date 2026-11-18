#!/usr/bin/env bash
# Live desk status without account figures.
cd "C:/Users/Asus/Desktop/AI_Trader" && AI_TRADER_ENVIRONMENT=mainnet target/debug/atx.exe live status 2>&1 | grep -v -iE "balance|equity|wallet|available"
