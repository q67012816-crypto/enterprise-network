"""DEV_RESULTS 31 (post hoc, descriptive): share of positives whose full-candidate rank lies in the top q% of the
investor's eligible candidates, logistic regression M1 vs M1 + interlocks. Reads out_fullcand/scores_*.csv only
(candidate set as corrected in DEV_RESULTS 28.2: extension rows only). Output: out_fullcand/coverage_topk.json."""
import collections, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import rd

L = Path(__file__).resolve().parents[1]; DX = L / 'data_v3x'; OF = L / 'out_fullcand'
VAL = (2021, 2022, 2023); QS = (.05, .10, .20, .33)
deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t in VAL for g in [rd(DX / f'graph_{t}.csv')]}

def per_pos(v):
    S = pd.read_csv(OF / f'scores_{v}.csv', dtype={'pair_id': str, 'anchor': str, 'other': str})
    r = {}
    for pid, g in S.groupby('pair_id'):
        p = g[g.label == 1].iloc[0]; n = g[(g.split == 'fullcand') & (g.label == 0)].score.values; s = p.score
        r[pid] = (p.fold, p.anchor, int(p.fy), min(deg[p.fy][p.anchor], deg[p.fy][p.other]), 1 + (n > s).sum() + .5 * (n == s).sum(), len(n))
    return pd.DataFrame(r, index=['fold', 'anchor', 'fy', 'md', 'rank', 'ncand']).T

A, B = per_pos('M1'), per_pos('M1+il')
W = A[['fold', 'anchor', 'md', 'ncand']].copy(); W['md'] = W.md.astype(int); W['ncand'] = W.ncand.astype(float)
W['rM1'] = A['rank'].astype(float); W['rIL'] = B['rank'].reindex(W.index).astype(float)
cl = (W.fold + '-' + W.anchor).values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}
rng = np.random.default_rng(20261103); DR = [np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]) for _ in range(10000)]
GR = {'all': np.ones(len(W), bool), 'low': (W.md <= 1).values, 'iso': (W.md == 0).values, 'high': (W.md >= 2).values}
out = {'ncand_mean': round(float(W.ncand.mean()), 1), 'n': {g: int(m.sum()) for g, m in GR.items()}, 'coverage': {}}
for q in QS:
    c1 = (W.rM1 <= q * W.ncand).values.astype(float); c2 = (W.rIL <= q * W.ncand).values.astype(float); d = c2 - c1
    for g, m in GR.items():
        v = [d[i][m[i]].mean() for i in DR if m[i].any()]
        out['coverage'][f'{g}|top{round(q * 100)}%'] = dict(random=q, M1=round(float(c1[m].mean()), 3), M1_il=round(float(c2[m].mean()), 3),
                                                           diff=round(float(d[m].mean()), 3), ci95=[round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)])
(OF / 'coverage_topk.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
for k, v in out['coverage'].items(): print(k, v)
print(out['n'], out['ncand_mean'])
