#!/usr/bin/env bash
# OOM-Schutz für Matrix 3: MemAvailable < 2 GiB -> bench-srv beenden, bevor der Kernel-OOM Systemdienste trifft.
# 6 GiB war zu streng (Ornith läuft regulär mit 5-6 GiB frei). Endet mit run_matrix3.sh.
L=$1; MAIN=$2
while kill -0 $MAIN 2>/dev/null; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if (( a < 2 )) && docker ps --format '{{.Names}}' | grep -qx bench-srv; then
    echo "OOM-SCHUTZ $(date +%T): MemAvailable ${a} GiB, bench-srv beendet" | tee -a $L; docker kill bench-srv >/dev/null 2>&1; fi
  sleep 1
done
