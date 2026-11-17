#!/usr/bin/env bash
# Stop every desk process started by desk_start.sh (process trees by recorded msys PID). Never touches STOCK SCANNER.
cd "C:/Users/Asus/Desktop/AI_Trader"
for f in logs/run/*.pid; do p=$(cat $f); w=$(cat /proc/$p/winpid 2>/dev/null); [ -n "$w" ] && taskkill //T //F //PID $w >/dev/null 2>&1; kill $p 2>/dev/null; echo "stopped $(basename $f .pid) ($p/$w)"; rm -f $f; done
