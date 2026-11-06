#!/bin/bash
# 第 18 节 C1–C3 的完整运行顺序（韩国数据先做，Photo 最后做）。日志：bench/out/c/run.log
cd "/d/claude data/enterprise-network/linkpred"
PY="/d/claude data/venvs/enet/Scripts/python.exe"
LOG=bench/out/c/run.log; mkdir -p bench/out/c
f() { "$PY" -u "$@" 2>&1 | grep -v "Warn\|warn\|return func" >> $LOG; }
echo "start $(date)" >> $LOG
# 韩国数据：C1（边丢弃）、C2（Cold Brew 两阶段）、C3（SEAL 调参 + 10 种子）
for v in dfgt-ed10 dfgt-ed30 sage-ed10 sage-ed30 coldbrew2; do f src/paper_dev.py run --variant $v --seeds 10 --device cuda --out out_dev_gpu10; done
f src/paper_dev.py tune-base --variant seal --seeds 3 --device cuda --out out_dev_gpu10
f src/paper_dev.py run --variant seal --seeds 10 --device cuda --out out_dev_gpu10
for d in Cora CiteSeer Photo; do
  for r in 0.1 0.2 0.3 0.4 0.5; do f src/bench.py run-c --data $d --r $r --models dfgt-ed10,dfgt-ed30,sage-ed10,sage-ed30; done
  for r in 0.0 0.1 0.2 0.3 0.4 0.5; do f src/bench.py run-c --data $d --r $r --models gae,coldbrew,coldbrew2; done
  for r in 0.0 0.1 0.2 0.3 0.4 0.5; do f src/bench.py run-c --data $d --r $r --models degfair,tailgnn; done
  for r in 0.1 0.3 0.5; do f src/bench.py run-seal --data $d --r $r; done
done
echo "ALL DONE $(date)" >> $LOG
