"""Unit checks + M0-M2 per protocol.md. Selection on validation; test evaluated once at the end."""
import collections, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
L = Path(__file__).resolve().parents[1]; D = L / 'data'; O = L / 'out'; O.mkdir(exist_ok=True)
X = pd.read_csv(D / 'pairs.csv', dtype={'anchor': str, 'other': str})
N = pd.read_csv(D / 'nodes.csv', dtype={'corp_code': str, 'industry': str, 'market': str, 'fin_disc': str}, keep_default_na=False, na_values=[''])
Hh = pd.read_csv(D / 'history_undirected.csv', dtype=str)
G = {t: pd.read_csv(D / f'graph_{t}.csv', dtype=str) for t in range(2021, 2026)}
und = lambda g: {frozenset(p) for p in zip(g.src, g.dst)}
# ---- unit checks
H = {t: {frozenset(p) for p in zip(g.a, g.b)} for t, g in Hh.groupby(Hh.fy.astype(int))}
chk = {}
fd = N.dropna(subset=['fin_disc']); chk['features_fin_disc_le_C'] = bool((fd.fin_disc.astype(str) <= (fd.fy + 1).astype(str) + '0430').all())
pos = X[X.label == 1]; neg = X[X.label == 0]
chk['positives_absent_from_history'] = all(frozenset((a, o)) not in H[t] for t, a, o in zip(pos.fy, pos.anchor, pos.other))
chk['negatives_not_history_or_next'] = all(frozenset((a, o)) not in H[t] and frozenset((a, o)) not in und(G[t + 1]) for t, a, o in zip(neg.fy, neg.anchor, neg.other)) if False else None
bad = 0
for t, g in neg.groupby('fy'):
    B = H[t] | und(G[t + 1]); bad += sum(frozenset((a, o)) in B for a, o in zip(g.anchor, g.other))
chk['negatives_not_history_or_next'] = bad == 0
chk['positives_in_next_graph'] = all(frozenset((a, o)) in und(G[t + 1]) for t, a, o in zip(pos.fy, pos.anchor, pos.other))
chk['negatives_per_positive'] = X.groupby('pair_id').label.agg(lambda s: (s == 0).sum()).eq(100).all()
chk = {k: bool(v) for k, v in chk.items()}; print(chk); assert all(chk.values())
# ---- features
FI = ['log_assets', 'liabilities_assets', 'equity_assets', 'capital_assets', 'revenue_assets', 'operating_income_assets', 'net_income_assets', 'signed_log_revenue', 'signed_log_operating_income', 'current_operating_loss']
NI = N.set_index(['fy', 'corp_code'])
tr_nodes = N[N.fy.isin([2021, 2022])]
lo, hi = tr_nodes[FI].quantile(.005), tr_nodes[FI].quantile(.995); med = tr_nodes[FI].clip(lo, hi, axis=1).median()
topind = tr_nodes.industry.value_counts().index[:20].tolist()
adj = {t: collections.defaultdict(set) for t in range(2021, 2025)}
for t in adj:
    for a, b in zip(G[t].src, G[t].dst): adj[t][a].add(b); adj[t][b].add(a)
def heur(t, a, o):
    A_, B_ = adj[t].get(a, set()), adj[t].get(o, set()); cn = A_ & B_
    return len(cn), sum(1 / np.log(len(adj[t][z])) for z in cn if len(adj[t][z]) > 1), len(cn) / len(A_ | B_) if A_ | B_ else 0., len(A_) * len(B_)
def design(df):
    out = {}
    for side, col in (('a', 'anchor'), ('o', 'other')):
        idx = list(zip(df.fy, df[col])); M = NI.reindex(idx)
        f = M[FI].reset_index(drop=True)
        for c in FI: out[f'{side}_miss_{c}'] = f[c].isna().astype(float).values; out[f'{side}_{c}'] = f[c].clip(lo[c], hi[c]).fillna(med[c]).values
        ind = M.industry.fillna('').values; mk = M.market.fillna('').values
        for i in topind: out[f'{side}_ind_{i}'] = (ind == i).astype(float)
        out[f'{side}_ind_other'] = (~np.isin(ind, topind)).astype(float)
        for m in ('Y', 'K'): out[f'{side}_mkt_{m}'] = (mk == m).astype(float)
        out[f'_{side}_ind'] = ind; out[f'_{side}_mk'] = mk
    out['same_industry'] = (out['_a_ind'] == out['_o_ind']).astype(float); out['same_market'] = (out['_a_mk'] == out['_o_mk']).astype(float)
    h = np.array([heur(t, a, o) for t, a, o in zip(df.fy, df.anchor, df.other)], dtype=float).reshape(-1, 4)
    out['cn'], out['aa'], out['jac'], out['pa'] = h.T
    return pd.DataFrame({k: v for k, v in out.items() if not k.startswith('_')}, index=df.index)
Z = design(X); M1C = [c for c in Z if c not in ('cn', 'aa', 'jac', 'pa')]
Z['l_cn'], Z['l_pa'] = np.log1p(Z.cn), np.log1p(Z.pa); M2C = M1C + ['l_cn', 'aa', 'jac', 'l_pa']
tr, va, te = (X.split == s for s in ('train', 'validation', 'test')); y = X.label.values
def fit(cols, C):
    mu, sd = Z.loc[tr, cols].mean(), Z.loc[tr, cols].std().replace(0, 1)
    m = LogisticRegression(C=C, max_iter=5000).fit((Z.loc[tr, cols] - mu) / sd, y[tr]); return lambda W: m.predict_proba((W[cols] - mu) / sd)[:, 1]
sel = {}; val = {}
for name, cols in (('M1', M1C), ('M1+scores', M2C)):
    r = {C: AP(y[va], fit(cols, C)(Z[va])) for C in (0.01, 0.1, 1, 10)}; best = max(r, key=r.get); sel[name] = best; val[name] = r
rng0 = np.random.default_rng(20261103)
def scorers():
    s = {'M0_random': None, 'M0_same_industry': lambda W: W.same_industry.values, 'M0_same_market': lambda W: W.same_market.values,
         'CN': lambda W: W.cn.values, 'AA': lambda W: W.aa.values, 'Jaccard': lambda W: W.jac.values, 'PA': lambda W: W.pa.values,
         'M1': fit(M1C, sel['M1']), 'M1+scores': fit(M2C, sel['M1+scores'])}
    return s
S = scorers(); X['rnd'] = rng0.random(len(X)); S['M0_random'] = lambda W: X.loc[W.index, 'rnd'].values
valAP = {k: AP(y[va], f(Z[va])) for k, f in S.items()}
bestH = max(['CN', 'AA', 'Jaccard', 'PA'], key=valAP.get)
frozen = dict(selected_C=sel, val_grid=val, val_AP=valAP, best_heuristic=bestH)
json.dump(frozen, open(O / 'validation_selection.json', 'w'), indent=1, default=float); print(json.dumps(frozen, indent=1, default=float))
# ---- ranking (full candidate ranking per investor)
Gall = {t: und(G[t + 1]) for t in range(2021, 2025)}
elig = {t: set(N[N.fy == t].corp_code) for t in range(2021, 2025)}
def ranking(split_mask, t):
    P = X[split_mask & (X.label == 1)]; Eset = sorted(elig[t] & set(pd.read_csv(D / 'nodes.csv', dtype={'corp_code': str}).query('fy==@t').corp_code))
    rows = []
    gn = pd.read_csv(D / f'graph_{t+1}.csv', dtype=str); E1 = set(N[N.fy == t].corp_code)
    for a, g in P.groupby('anchor'):
        cand = [w for w in Eset if w != a and frozenset((a, w)) not in H[t]]
        C_ = pd.DataFrame({'fy': t, 'anchor': a, 'other': cand}); rows.append(C_.assign(is_pos=C_.other.isin(set(g.other)).astype(int)))
    R = pd.concat(rows, ignore_index=True)
    # restrict to candidates eligible also at t+1 (protocol §1)
    return R
res = {}
def evalset(mask, tag):
    W = Z[mask]; yy = y[mask]; out = {}
    for k, f in S.items(): out[k] = f(W)
    return out
def metrics(scores, mask):
    yy = y[mask]; return {k: dict(AP=AP(yy, s), AUC=AUC(yy, s)) for k, s in scores.items()}
# eligibility at t+1 for ranking candidates
N1 = {t: set(pd.read_csv(D / 'graph_years.csv').fy) for t in []}
def rank_metrics(t, mask):
    R = ranking(mask, t)
    nxt = set(X[(X.fy == t)].anchor) | set(X[(X.fy == t)].other)  # nodes eligible both years appear in pairs; use eligibility file instead
    elig_next = set(pd.read_csv(D / 'eligible.csv', dtype={'corp_code': str}).query('fy==@t+1').corp_code)
    R = R[R.other.isin(elig_next)].copy(); R.index = range(len(R))
    R['fy'] = t; RZ = design(R); RZ['l_cn'], RZ['l_pa'] = np.log1p(RZ.cn), np.log1p(RZ.pa)
    rr = np.random.default_rng(20261103); out = {}
    for k, f in S.items():
        s = rr.random(len(R)) if k == 'M0_random' else f(RZ)
        R['s'] = s; h10 = h50 = mrr = []; H10, H50, MRR, PRR = [], [], [], []
        for a, g in R.groupby('anchor'):
            sv, pv = g.s.values, g.is_pos.values.astype(bool)
            def er(x, pool): return (pool > x).sum() + ((pool == x).sum() + 1) / 2
            ranks = np.array([er(x, sv) for x in sv[pv]]); best = ranks.min()
            H10.append(best <= 10); H50.append(best <= 50); MRR.append(1 / best)
            negs = sv[~pv]; PRR += [1 / ((negs > x).sum() + ((negs == x).sum() + 1) / 2) for x in sv[pv]]
        out[k] = dict(Hits10=np.mean(H10), Hits50=np.mean(H50), MRR=np.mean(MRR), filtered_RR_per_positive=np.mean(PRR), n_investors=len(H10), n_candidates=len(R))
    return out
import sys
if __name__ == '__main__':
    pd.DataFrame([dict(k=k, v=v) for k, v in chk.items()]).to_csv(O / 'unit_checks.csv', index=False)
    valm = {'all': metrics(evalset(va, 'v'), va)}
    json.dump(dict(val_metrics=valm, val_ranking=rank_metrics(2023, va)), open(O / 'validation_metrics.json', 'w'), indent=1, default=float)
    if '--test' not in sys.argv: sys.exit(0)
    # ===== TEST: evaluated once =====
    assert not (O / 'test_frozen.json').exists(), 'test already evaluated once'
    Tm = te; sc = {k: f(Z[Tm]) for k, f in S.items()}; T = X[Tm].copy()
    for k, s in sc.items(): T[k] = s
    T.to_csv(O / 'test_scores.csv', index=False)
    cells = {'all': np.ones(len(T), bool)}
    subset_ids = set(T[(T.label == 1) & T.subset_nonfin_ge5].pair_id); cells['nonfin_ge5'] = T.pair_id.isin(subset_ids).values
    for gname in ('both_connected', 'one_isolated', 'both_isolated'):
        cells[f'group_{gname}'] = (T.group == gname).values; cells[f'nonfin_ge5_group_{gname}'] = cells['nonfin_ge5'] & cells[f'group_{gname}']
    rng = np.random.default_rng(20261103); out = {}
    comps = [(bestH, 'M1'), ('M1+scores', 'M1')] + [(h, 'M1') for h in ['CN', 'AA', 'Jaccard', 'PA'] if h != bestH]
    for cname, m in cells.items():
        U = T[m]; npos = int(U.label.sum()); e = dict(n_pairs=int(m.sum()), n_pos=npos, informative=npos >= 30)
        if npos == 0 or npos == len(U): out[cname] = e; continue
        e['AP'] = {k: AP(U.label, U[k]) for k in S}; e['AUC'] = {k: AUC(U.label, U[k]) for k in S}
        an = U.anchor.values; ua = np.unique(an); ix = {a: np.where(an == a)[0] for a in ua}
        bs = {f'{a}-{b}': [] for a, b in comps}
        for _ in range(1000):
            idx = np.concatenate([ix[a] for a in rng.choice(ua, len(ua))]); V_ = U.iloc[idx]
            if V_.label.nunique() < 2: continue
            for a, b in comps: bs[f'{a}-{b}'].append(AP(V_.label, V_[a]) - AP(V_.label, V_[b]))
        e['diff'] = {k: dict(point=e['AP'][k.split('-M1')[0]] - e['AP']['M1'], lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), n_boot=len(v)) for k, v in bs.items()}
        out[cname] = e
    out['test_ranking'] = rank_metrics(2024, te)
    json.dump(out, open(O / 'test_frozen.json', 'w'), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
