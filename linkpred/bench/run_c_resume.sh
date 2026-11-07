#!/bin/bash
# 接续 run_c.sh：2026-11-07 21:0x 按 Mingsen 要求暂停后剩下的格子（Photo 的 Tail-GNN r=0.1–0.5、DegFairGNN r=0.2–0.5、SEAL）。
cd "/d/claude data/enterprise-network/linkpred"
PY="/d/claude data/venvs/enet/Scripts/python.exe"
LOG=bench/out/c/run.log
f() { "$PY" -u "$@" 2>&1 | grep -v "Warn\|warn\|return func" >> $LOG; }
echo "resume $(date)" >> $LOG
f src/bench.py run-c --data Photo --r 0.1 --models tailgnn
for r in 0.2 0.3 0.4 0.5; do f src/bench.py run-c --data Photo --r $r --models degfair,tailgnn; done
for r in 0.1 0.3 0.5; do f src/bench.py run-seal --data Photo --r $r; done
echo "ALL DONE $(date)" >> $LOG
