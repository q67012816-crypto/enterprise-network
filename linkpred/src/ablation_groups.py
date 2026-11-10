"""DEV_RESULTS 第 27 节：按度分组的去偏组件消融与等价性检验（只用 out_dev_gpu10 已有分数；不改冻结代码）。
用法：python src/ablation_groups.py   # 写 out_dev_rewire/ablation_groups.json"""
import collections, json
from pathlib import Path
import numpy as np, pandas as pd
L = Path(__file__).resolve().parents[1]
deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t in (2021, 2022, 2023) for g in [pd.read_csv(L / f'data_v3x/graph_{t}.csv', dtype=str)]}
SC = [f'score_s{i}' for i in range(1, 11)]

def per_pos(name):
    """每个正例：fold、anchor、较小端度数、集成 AUC、逐种子 AUC（10 列）。"""
    S = pd.read_csv(L / f'out_dev_gpu10/scores_{name}.csv', dtype={'pair_id': str, 'anchor': str, 'other': str})
    r = {}
    for pid, g in S.groupby('pair_id'):
        p = g[g.label == 1].iloc[0]; neg = g[g.label == 0]
        auc = lambda c: ((neg[c].values < p[c]).sum() + .5 * (neg[c].values == p[c]).sum()) / len(neg)
        r[pid] = (p.fold, p.anchor, min(deg[p.fy][p.anchor], deg[p.fy][p.other]), auc('score'), *[auc(c) for c in SC])
    return pd.DataFrame(r, index=['fold', 'anchor', 'md', 'ens', *SC]).T

NAMES = ['dfgt', 'dfgt-noaug', 'dfgt-nosim', 'dfgt-nossl', 'dfgt0', 'dfgt-nograph']
P = {n: per_pos(n) for n in NAMES}
W = P['dfgt'][['fold', 'anchor', 'md']].copy(); W['md'] = W.md.astype(int); W['cl'] = W.fold + '-' + W.anchor
cl = W.cl.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}
rng = np.random.default_rng(20261103)
DR = [np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]) for _ in range(10000)]
GR = {'d0': (W.md == 0).values, 'd1': (W.md == 1).values, 'low': (W.md <= 1).values, 'high': (W.md >= 2).values, 'all': np.ones(len(W), bool)}

def ci(d, m):
    v = [d[i][m[i]].mean() for i in DR if m[i].any()]
    lo95, hi95, lo90, hi90 = (np.percentile(v, q) for q in (2.5, 97.5, 5, 95))
    est = float(d[m].mean())
    verdict = ('等价于零' if -.02 <= lo90 and hi90 <= .02 else ('有贡献（%s）' % ('正' if est > 0 else '负') if lo95 > 0 or hi95 < 0 else '检验不出'))
    return dict(est=round(est, 4), ci95=[round(float(lo95), 4), round(float(hi95), 4)], ci90=[round(float(lo90), 4), round(float(hi90), 4)],
                mde=round(float(hi95 - lo95) / 2, 4), n=int(m.sum()), verdict=verdict)

COMP = [('aug', 'dfgt', 'dfgt-noaug'), ('sim', 'dfgt', 'dfgt-nosim'), ('ssl', 'dfgt', 'dfgt-nossl'), ('zero_k5', 'dfgt0', 'dfgt'), ('graph', 'dfgt', 'dfgt-nograph')]
res = dict(counts={g: int(m.sum()) for g, m in GR.items()}, margin=.02, B=10000)
for key, a, b in COMP:
    da = (P[a]['ens'].astype(float) - P[b]['ens'].astype(float)).values
    db = np.mean([(P[a][c].astype(float) - P[b][c].astype(float)).values for c in SC], axis=0)
    res[key] = {'ensemble': {g: ci(da, m) for g, m in GR.items()}, 'seed_paired': {g: ci(db, m) for g, m in GR.items()}}
json.dump(res, open(L / 'out_dev_rewire/ablation_groups.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False, indent=1))
