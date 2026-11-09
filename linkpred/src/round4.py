"""DEV_RESULTS 第 26.1 节：效应换算（排序指标）与年份异质性（只用开发年已有分数）。
用法：python src/round4.py   # 写 out_dev_rewire/round4_eval.json"""
import collections, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2
L = Path(__file__).resolve().parents[1]
deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t in (2021, 2022, 2023) for g in [pd.read_csv(L / f'data_v3x/graph_{t}.csv', dtype=str)]}
def per_pos(f):
    S = pd.read_csv(f, dtype={'pair_id': str, 'anchor': str, 'other': str}); r = {}
    for pid, g in S.groupby('pair_id'):
        p = g[g.label == 1].iloc[0]; n = g.score.values[g.label.values == 0]; s = p.score
        rk = 1 + (n > s).sum() + .5 * (n == s).sum()
        r[pid] = (p.fold, p.anchor, int(p.fy), min(deg[p.fy][p.anchor], deg[p.fy][p.other]), ((n < s).sum() + .5 * (n == s).sum()) / len(n), rk)
    return pd.DataFrame(r, index=['fold', 'anchor', 'fy', 'md', 'auc', 'rank']).T
F = {'M1': L / 'out_dev_rewire/real/scores_M1.csv', 'M1+il': L / 'out_dev_rewire/real/scores_M1+il.csv', 'dfgt': L / 'out_dev_gpu10/scores_dfgt.csv', 'M1_gpu10': L / 'out_dev_gpu10/scores_M1.csv'}
P = {k: per_pos(f) for k, f in F.items()}
W = P['M1'][['fold', 'anchor', 'fy', 'md']].copy(); W['md'] = W.md.astype(int); W['fy'] = W.fy.astype(int); W['cl'] = W.fold + '-' + W.anchor
MET = {'H10': lambda r: (r <= 10).astype(float), 'H1': lambda r: (r <= 1).astype(float), 'rank': lambda r: r, 'rr': lambda r: 1 / r}
for k in P:
    r = P[k]['rank'].reindex(W.index).astype(float)
    for m, f in MET.items(): W[f'{k}|{m}'] = f(r)
    W[f'{k}|auc'] = P[k].auc.reindex(W.index).astype(float)
GR = lambda U: {'low': (U.md <= 1).values, 'iso': (U.md == 0).values, 'all': np.ones(len(U), bool)}
def draws(U, B=1000, seed=20261103):
    cl = U.cl.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(seed)
    return [np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]) for _ in range(B)]
DR = draws(W)
def ci(d, m, dr=DR):
    v = [d[i][m[i]].mean() for i in dr if m[i].any()]; return [round(float(d[m].mean()), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
res = {'levels': {}, 'diff': {}, 'median_rank': {}}
for g, m in GR(W).items():
    res['levels'][g] = {k: {mm: round(float(W.loc[m, f'{k}|{mm}'].mean()), 4) for mm in ('auc', 'H10', 'H1', 'rank', 'rr')} for k in P}
    res['median_rank'][g] = {k: float(W.loc[m, f'{k}|rank'].median()) for k in P}
    for a, b in (('M1+il', 'M1'), ('dfgt', 'M1_gpu10')):
        res['diff'].setdefault(f'{a} − {b}', {})[g] = {mm: ci((W[f'{a}|{mm}'] - W[f'{b}|{mm}']).values, m) for mm in ('auc', 'H10', 'H1', 'rank', 'rr')}
# 年份异质性（M1+il − M1 的 AUC 增益）
het = {}
for g in ('low', 'iso', 'all'):
    est, se, dist = {}, {}, {}
    for t in (2021, 2022, 2023):
        U = W[W.fy == t]; m = GR(U)[g]; d = (U['M1+il|auc'] - U['M1|auc']).values; dr = draws(U)
        v = np.array([d[i][m[i]].mean() for i in dr if m[i].any()]); est[t] = float(d[m].mean()); se[t] = float(v.std(ddof=1)); dist[t] = v[:min(len(v), 1000)]
    w = {t: 1 / se[t] ** 2 for t in est}; mu = sum(w[t] * est[t] for t in est) / sum(w.values()); Q = sum(w[t] * (est[t] - mu) ** 2 for t in est)
    pair = {}
    for a, b in ((2022, 2021), (2023, 2021), (2023, 2022)):
        n = min(len(dist[a]), len(dist[b])); dd = dist[a][:n] - dist[b][:n]
        pair[f'{a}−{b}'] = [round(est[a] - est[b], 4), round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)]
    het[g] = dict(est={t: round(x, 4) for t, x in est.items()}, se={t: round(x, 4) for t, x in se.items()}, pooled_fixed=round(mu, 4), Q=round(Q, 3), df=2, p=round(float(chi2.sf(Q, 2)), 4), pairwise=pair)
res['heterogeneity'] = het
json.dump(res, open(L / 'out_dev_rewire/round4_eval.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False); print(json.dumps(res, ensure_ascii=False, indent=1))
