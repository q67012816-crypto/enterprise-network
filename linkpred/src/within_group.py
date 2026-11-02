"""POST-HOC (added after viewing 2024->2025 test): within-candidate-set metrics. Each positive ranked among its own
101 candidates (1 pos + 100 negs, same pair_id); ties by expected rank (A1). Cluster bootstrap by investor, 1000, seed 20261103."""
import json, sys
import numpy as np, pandas as pd
out_dir, fname, bestH = sys.argv[1], sys.argv[2], sys.argv[3]
T = pd.read_csv(f'{out_dir}/{fname}', dtype={'anchor': str, 'other': str})
MODELS = ['M0_random', 'M0_same_industry', 'M0_same_market', 'CN', 'AA', 'Jaccard', 'PA', 'M1', 'M1+scores']
rows = []
for pid, g in T.groupby('pair_id', sort=False):
    p = g[g.label == 1].iloc[0]; n = g[g.label == 0]; r = dict(pair_id=pid, anchor=p.anchor, group=p.group, sub=bool(p.subset_nonfin_ge5))
    for m in MODELS:
        x, ns = p[m], n[m].values; gt, eq = (ns > x).sum(), (ns == x).sum(); rank = gt + 1 + eq / 2
        r[f'{m}|AUC'] = ((ns < x).sum() + eq / 2) / len(ns); r[f'{m}|RR'] = 1 / rank; r[f'{m}|H10'] = float(rank <= 10)
    rows.append(r)
W = pd.DataFrame(rows); rng = np.random.default_rng(20261103)
comps = [('M1+scores', 'M1'), (bestH, 'M1'), ('M1+scores', bestH)]
cells = {'all': W.index == W.index, 'nonfin_ge5': W['sub'].values}
for gname in ('both_connected', 'one_isolated', 'both_isolated'): cells[gname] = (W.group == gname).values
res = {}
for c, m in cells.items():
    U = W[m]; e = dict(n_pos=int(m.sum()), informative=bool(m.sum() >= 30))
    if len(U) == 0: res[c] = e; continue
    e['mean'] = {f'{mod}|{k}': U[f'{mod}|{k}'].mean() for mod in MODELS for k in ('AUC', 'RR', 'H10')}
    an = U.anchor.values; ua = np.unique(an); ix = {a: np.where(an == a)[0] for a in ua}
    B = {f'{a}-{b}|{k}': [] for a, b in comps for k in ('AUC', 'RR', 'H10')}
    for _ in range(1000):
        idx = np.concatenate([ix[a] for a in rng.choice(ua, len(ua))]); V = U.iloc[idx]
        for a, b in comps:
            for k in ('AUC', 'RR', 'H10'): B[f'{a}-{b}|{k}'].append(V[f'{a}|{k}'].mean() - V[f'{b}|{k}'].mean())
    e['diff'] = {key: dict(point=U[f"{key.split('-')[0]}|{key.split('|')[1]}"].mean() - U[f"{key.split('-')[1].split('|')[0]}|{key.split('|')[1]}"].mean(),
                           lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5))) for key, v in B.items()}
    res[c] = e
W.to_csv(f'{out_dir}/within_{fname}', index=False)
json.dump(res, open(f'{out_dir}/within_{fname.replace(".csv", ".json")}', 'w'), indent=1)
for c, e in res.items():
    print(c, e['n_pos'], e['informative'])
    if 'mean' in e:
        print('  ', {m: tuple(round(e['mean'][f'{m}|{k}'], 3) for k in ('AUC', 'RR', 'H10')) for m in MODELS})
        print('  ', {k: (round(v['point'], 3), round(v['lo'], 3), round(v['hi'], 3)) for k, v in e['diff'].items()})
