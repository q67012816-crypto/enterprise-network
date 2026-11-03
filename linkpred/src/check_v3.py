"""data_v3 的单元检查与汇总（只读，不训练模型）。
用法：python src/check_v3.py [--data data_v3] [--ref data_a2_apionly]
输出：<data>/unit_checks.csv、<data>/summary.json，并在终端打印 v3 与参照版本的对比表。"""
import argparse, collections, json
from pathlib import Path
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--data', default='data_v3'); ap.add_argument('--ref', default='data_a2_apionly'); ARGS = ap.parse_args()
L = Path(__file__).resolve().parents[1]; D, REF = L / ARGS.data, L / ARGS.ref
rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
und = lambda g: {frozenset(p) for p in zip(g.src, g.dst)}

def load(d):
    """读取一个数据版本。输入：目录；输出：(点对表, 各年无向图, 各年历史集合, 节点表, 合格名单)。"""
    X = rd(d / 'pairs.csv'); X['fy'] = X.fy.astype(int); X['label'] = X.label.astype(int)
    G = {t: und(rd(d / f'graph_{t}.csv')) for t in range(2019, 2026)}
    Hh = rd(d / 'history_undirected.csv'); H = {int(t): {frozenset(p) for p in zip(g.a, g.b)} for t, g in Hh.groupby('fy')}
    N = rd(d / 'nodes.csv'); N['fy'] = N.fy.astype(int)
    E = rd(d / 'eligible.csv'); E['fy'] = E.fy.astype(int)
    return X, G, H, N, E

def unit_checks(X, G, H, N, E):
    """检查泄漏与构造规则。输入：load() 的结果；输出：{检查名: 是否通过}。"""
    pos, neg = X[X.label == 1], X[X.label == 0]; elig = {t: set(g.corp_code) for t, g in E.groupby('fy')}
    pre = rd(D / 'preexisting_pairs.csv') if (D / 'preexisting_pairs.csv').exists() else pd.DataFrame(columns=['fy', 'a', 'b'])
    PRE = {(int(t), frozenset((a, b))) for t, a, b in zip(pre.fy, pre.a, pre.b)}
    fd = N[N.fin_disc.ne('')]
    c = {}
    c['features_fin_disc_le_C'] = bool((fd.fin_disc <= (fd.fy + 1).astype(str) + '0430').all())
    c['positives_absent_from_history_Ht'] = all(frozenset((a, o)) not in H[t] for t, a, o in zip(pos.fy, pos.anchor, pos.other))
    c['positives_in_next_graph'] = all(frozenset((a, o)) in G[t + 1] for t, a, o in zip(pos.fy, pos.anchor, pos.other))
    c['positives_not_in_feature_graph_Gt'] = all(frozenset((a, o)) not in G[t] for t, a, o in zip(pos.fy, pos.anchor, pos.other))
    c['negatives_not_in_Ht_or_Gt1'] = all(frozenset((a, o)) not in H[t] and frozenset((a, o)) not in G[t + 1] for t, a, o in zip(neg.fy, neg.anchor, neg.other))
    c['preexisting_not_positive'] = not any((t, frozenset((a, o))) in PRE for t, a, o in zip(pos.fy, pos.anchor, pos.other))
    c['preexisting_not_negative'] = not any((t, frozenset((a, o))) in PRE for t, a, o in zip(neg.fy, neg.anchor, neg.other))
    c['endpoints_eligible_t_and_t1'] = all(a in elig[t] and o in elig[t] and a in elig[t + 1] and o in elig[t + 1] for t, a, o in zip(X.fy, X.anchor, X.other))
    c['exactly_100_negatives'] = bool(X.groupby('pair_id').label.agg(lambda s: (s == 0).sum()).eq(100).all())
    if 'new_evidence' in X: c['every_positive_has_new_evidence'] = bool(pos.new_evidence.isin(['new_opening_zero', 'new_acquired_in_year']).all())
    return c

def counts(X, G, E):
    """各年正例数、分组、子集和孤立比例。输入：点对表、图、合格名单；输出：DataFrame（每年一行）。"""
    rows = []
    for t, g in X[X.label == 1].groupby('fy'):
        el = set(E[E.fy == t].corp_code); conn = {n for p in G[t] for n in p}
        r = dict(fy=t, split=g.split.iloc[0], positives=len(g), both_connected=int((g.group == 'both_connected').sum()),
                 one_isolated=int((g.group == 'one_isolated').sum()), both_isolated=int((g.group == 'both_isolated').sum()),
                 nonfin_ge5=int((g.subset_nonfin_ge5 == 'True').sum()), financial_investor=int((g.investor_financial == 'True').sum()),
                 edges_Gt=len(G[t]), isolated_share_Gt=round(len(el - conn) / len(el), 4))
        if 'new_evidence' in g: r['flag_acquired_in_year'] = int((g.new_evidence == 'new_acquired_in_year').sum())
        rows.append(r)
    return pd.DataFrame(rows)

X, G, H, N, E = load(D); chk = unit_checks(X, G, H, N, E)
pd.DataFrame([dict(check=k, passed=v) for k, v in chk.items()]).to_csv(D / 'unit_checks.csv', index=False)
cv3 = counts(X, G, E); Xr, Gr, Hr, Nr, Er = load(REF); cref = counts(Xr, Gr, Er)
# 正例集合的变化：v2 正例在 v3 中的去向
pv2 = {(t, frozenset((a, o))) for t, a, o in zip(Xr[Xr.label == 1].fy, Xr[Xr.label == 1].anchor, Xr[Xr.label == 1].other)}
pv3 = {(t, frozenset((a, o))) for t, a, o in zip(X[X.label == 1].fy, X[X.label == 1].anchor, X[X.label == 1].other)}
pre = rd(D / 'preexisting_pairs.csv'); PRE = {(int(t), frozenset((a, b))) for t, a, b in zip(pre.fy, pre.a, pre.b)}
moves = collections.Counter()
for t, k in pv2:
    if (t, k) in pv3: moves[(t, 'kept')] += 1
    elif (t, k) in PRE: moves[(t, 'now preexisting')] += 1
    elif k in H[t] or k in G[t]: moves[(t, 'now in history/G_t (name fix)')] += 1
    else: moves[(t, 'other')] += 1
for t, k in pv3 - pv2: moves[(t, 'new in v3')] += 1
mv = pd.Series(moves).unstack(fill_value=0)
json.dump(dict(unit_checks=chk, counts_v3=cv3.to_dict('records'), counts_ref=cref.to_dict('records'), positive_moves=mv.to_dict('index')),
          open(D / 'summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=int)
print(chk); print('--- v3'); print(cv3.to_string(index=False)); print('---', ARGS.ref); print(cref.to_string(index=False)); print('--- v2 positives in v3'); print(mv.to_string())
