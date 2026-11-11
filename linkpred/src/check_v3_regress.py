"""修订 A8 第 2.1 节的回归门槛：新构建的数据（例如 2027 年加入 t = 2025 后）在 fy ≤ --max-fy 的部分必须与参照版本（data_v3x）完全一致，
否则不得进入打分。不调用 API，不修改任何数据目录。
比较内容：
  - pairs.csv、preexisting_pairs.csv、nodes.csv：fy ≤ max-fy 的行，逐行文本一致（含列顺序和行顺序）；
  - graph_<t>.csv（t ≤ max-fy + 1，即参照版本已有的特征图）：逐字节一致；
  - history_undirected.csv：fy ≤ max-fy 的点对集合一致（这个文件的行顺序来自 Python 集合迭代，随进程的字符串哈希种子变化，内容不变）；
  - eligible.csv：fy ≤ max-fy + 1 的行一致。
用法：python src/check_v3_regress.py --new data_v3x_2027 --ref data_v3x --max-fy 2024
2026-11-08 的检查：用当前代码重建 data_v3x（--first-year 2019）到临时目录，与 data_v3x 比较，全部一致（max-fy 2024）。"""
import argparse, sys
from pathlib import Path
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--new', required=True); ap.add_argument('--ref', default='data_v3x'); ap.add_argument('--max-fy', type=int, default=2024)
A = ap.parse_args(); L = Path(__file__).resolve().parents[1]; N, R = L / A.new, L / A.ref
rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
res = {}
def rows(name, max_fy):
    a, b = rd(R / name), rd(N / name)
    a, b = a[a.fy.astype(int) <= max_fy].reset_index(drop=True), b[b.fy.astype(int) <= max_fy].reset_index(drop=True)
    return list(a.columns) == list(b.columns) and a.equals(b), len(a), len(b)
for name in ('pairs.csv', 'preexisting_pairs.csv', 'nodes.csv'): res[name] = rows(name, A.max_fy)
res['eligible.csv'] = rows('eligible.csv', A.max_fy + 1)
for g in sorted(R.glob('graph_2*.csv')):
    t = int(g.stem.split('_')[1])
    if t <= A.max_fy + 1: res[g.name] = ((N / g.name).exists() and (N / g.name).read_bytes() == g.read_bytes(), g.stat().st_size, (N / g.name).stat().st_size if (N / g.name).exists() else 0)
ha, hb = rd(R / 'history_undirected.csv'), rd(N / 'history_undirected.csv')
sa = {(r.fy, r.a, r.b) for r in ha.itertuples() if int(r.fy) <= A.max_fy}; sb = {(r.fy, r.a, r.b) for r in hb.itertuples() if int(r.fy) <= A.max_fy}
res['history_undirected.csv (as set)'] = (sa == sb, len(sa), len(sb))
ok = all(v[0] for v in res.values())
for k, (p, na, nb) in res.items(): print(f"{'OK  ' if p else 'FAIL'} {k:34s} ref {na}  new {nb}")
print('REGRESSION CHECK', 'PASSED' if ok else 'FAILED', f'(fy <= {A.max_fy})')
sys.exit(0 if ok else 1)
