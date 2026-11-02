"""M3 测试结果评估（探索性）：组内 AUC / MRR / Hits@10（主），混合池 AP（次）。
所有区间：按投资方（anchor）聚类的配对 bootstrap，1000 次，rng=default_rng(20261103)，
同一组重抽样同时用于所有模型与所有比较（配对）。只读已冻结的分数文件，不再训练或调参。"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP
L = Path(__file__).resolve().parents[1]; O = L / 'out_2_m3'
# 冻结的 M1、M1+scores、PA 测试分数（Stage 1C, A4 数据）+ M3 测试分数（只评估过一次）
T = pd.read_csv(L / 'out_1c_a4/test_scores.csv', dtype={'anchor': str, 'other': str})
M3 = pd.read_csv(O / 'test_scores_m3.csv', dtype={'anchor': str, 'other': str})
T = T.merge(M3[['pair_id', 'other', 'M3']], on=['pair_id', 'other'], validate='1:1'); assert T.M3.notna().all()
MODELS = ['M3', 'M1', 'M1+scores', 'PA']
COMPS = [('M3', 'M1'), ('M3', 'M1+scores'), ('M3', 'PA'), ('M1+scores', 'M1')]   # 第三项即 S2（PA 由 A6 选出）
# 每个正例的组内指标（101 个候选，并列取期望秩）
rows = []
for pid, g in T.groupby('pair_id', sort=False):
    p = g[g.label == 1].iloc[0]; n = g[g.label == 0]
    r = dict(pair_id=pid, anchor=p.anchor, group=p.group, sub=bool(p.subset_nonfin_ge5))
    for m in MODELS:
        x, ns = p[m], n[m].values; gt, eq = (ns > x).sum(), (ns == x).sum(); rank = gt + 1 + eq / 2
        r[f'{m}|AUC'] = ((ns < x).sum() + eq / 2) / len(ns); r[f'{m}|MRR'] = 1 / rank; r[f'{m}|H10'] = float(rank <= 10)
    rows.append(r)
W = pd.DataFrame(rows)
cells = {'all': np.ones(len(W), bool), 'nonfin_ge5': W['sub'].values}
for gname in ('both_connected', 'one_isolated', 'both_isolated'): cells[gname] = (W.group == gname).values
res = {}
for c, m in cells.items():
    U = W[m].reset_index(drop=True); e = dict(n_pos=int(m.sum()), informative=bool(m.sum() >= 30))
    # 该格子的混合池 AP：正例及其自身的 100 个负例
    TP = T[T.pair_id.isin(U.pair_id)].reset_index(drop=True)
    e['mean'] = {f'{mod}|{k}': float(U[f'{mod}|{k}'].mean()) for mod in MODELS for k in ('AUC', 'MRR', 'H10')}
    e['mean'].update({f'{mod}|AP': float(AP(TP.label, TP[mod])) for mod in MODELS})
    ua = np.unique(U.anchor); ixW = {a: np.where(U.anchor.values == a)[0] for a in ua}; ixT = {a: np.where(TP.anchor.values == a)[0] for a in ua}
    rng = np.random.default_rng(20261103); B = {f'{a}-{b}|{k}': [] for a, b in COMPS for k in ('AUC', 'MRR', 'H10', 'AP')}
    for _ in range(1000):
        draw = rng.choice(ua, len(ua))     # 重抽样投资方，其全部正例和负例一起带入
        V = U.iloc[np.concatenate([ixW[a] for a in draw])]; VT = TP.iloc[np.concatenate([ixT[a] for a in draw])]
        apv = {mod: AP(VT.label, VT[mod]) if VT.label.nunique() > 1 else np.nan for mod in MODELS}
        for a, b in COMPS:
            for k in ('AUC', 'MRR', 'H10'): B[f'{a}-{b}|{k}'].append(V[f'{a}|{k}'].mean() - V[f'{b}|{k}'].mean())
            B[f'{a}-{b}|AP'].append(apv[a] - apv[b])
    e['diff'] = {}
    for key, v in B.items():
        (pair, k) = key.split('|'); a, b = pair.split('-', 1)
        e['diff'][key] = dict(point=e['mean'][f'{a}|{k}'] - e['mean'][f'{b}|{k}'], lo=float(np.nanpercentile(v, 2.5)), hi=float(np.nanpercentile(v, 97.5)))
    res[c] = e
W.to_csv(O / 'within_test_m3.csv', index=False); json.dump(res, open(O / 'test_metrics_m3.json', 'w'), indent=1)
for c, e in res.items():
    print(f'== {c}  n_pos={e["n_pos"]}  informative={e["informative"]}')
    for mod in MODELS: print(f'  {mod:10s}', ' '.join(f'{k}={e["mean"][f"{mod}|{k}"]:.3f}' for k in ('AUC', 'MRR', 'H10', 'AP')))
    for key, d in e['diff'].items(): print(f'  {key:22s} {d["point"]:+.3f} [{d["lo"]:+.3f}, {d["hi"]:+.3f}]')
