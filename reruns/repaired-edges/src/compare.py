"""Compare graph variants vs original: degree/isolation changes, metrics, repaired-vs-original paired bootstrap, groups."""
import json, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as ROC
V = ['original', 'r2024doc', 'rall_api']; R = {}
rd = lambda v, f: pd.read_csv(f'out/{v}/{f}', dtype={'corp_code': str, 'src_corp_code': str, 'target': str})
pan = {v: rd(v, 'model-panel.csv').set_index(['corp_code', 'feature_fy']) for v in V}
out = {}
# node-level degree changes on full observed graph (undirected pooled)
def deg(v, y):
    e = rd(v, f'graphs/edges-{y}-asof-{y+1}0430.csv'); s = {}
    for a, b in zip(e.src_corp_code, e.target): s.setdefault(a, set()).add(b); s.setdefault(b, set()).add(a)
    return {k: len(x) for k, x in s.items()}
for v in V[1:]:
    for y in (2021, 2022, 2023, 2024):
        d0, d1 = deg('original', y), deg(v, y); ks = set(d0) | set(d1)
        out[f'{v}_nodes_degree_changed_{y}'] = sum(d0.get(k, 0) != d1.get(k, 0) for k in ks)
tab = []
for v in V:
    p = pan[v]
    for sp in ('train', 'validation', 'test'):
        q = p[p.split.eq(sp)]
        tab.append({'variant': v, 'split': sp, 'n': len(q), 'isolated': int(q.degree.eq(0).sum()), 'connected': int(q.degree.gt(0).sum()),
                    'connected_no_neighbor_financials': int((q.degree.gt(0) & q.neighbor_financial_count.eq(0)).sum()),
                    'rows_degree_changed_vs_original': int((q.degree != pan['original'].loc[q.index, 'degree']).sum())})
iso = pd.DataFrame(tab); iso.to_csv('out/isolation-summary.csv', index=False)
M = pd.concat([rd(v, 'metrics.csv').assign(variant=v) for v in V]); M.to_csv('out/metrics-all.csv', index=False)
G = pd.concat([rd(v, 'group-metrics.csv').assign(variant=v) for v in V]); G.to_csv('out/group-metrics-all.csv', index=False)
B = pd.concat([rd(v, 'paired-bootstrap.csv').assign(variant=v) for v in V]); B.to_csv('out/bootstrap-neighbors-vs-financial-all.csv', index=False)
# repaired minus original, same model; paired firms, 500 reps, rng 20261102 (one stream per variant, fixed order)
P = {v: rd(v, 'predictions.csv').query("split=='test'") for v in V}
test = pan['original'][pan['original'].split.eq('test')].reset_index(); y = test.label.to_numpy(); rows = []
for v in V[1:]:
    rng = np.random.default_rng(20261102); ixs = [rng.integers(0, len(y), len(y)) for _ in range(500)]
    for fam in ('logistic', 'hist_gradient_boosting'):
        for cond in ('financial', 'financial_degree', 'financial_degree_neighbors'):
            g = lambda w: P[w][(P[w].family == fam) & (P[w].features == cond)].set_index('corp_code').loc[test.corp_code, 'probability'].to_numpy()
            a, o = g(v), g('original')
            for n, f in (('AP', AP), ('ROC_AUC', ROC)):
                dd = [f(y[i], a[i]) - f(y[i], o[i]) for i in ixs if len(np.unique(y[i])) == 2]
                rows.append({'variant': v, 'family': fam, 'features': cond, 'metric': n, 'difference': f(y, a) - f(y, o),
                             'ci95_low': np.percentile(dd, 2.5), 'ci95_high': np.percentile(dd, 97.5), 'reps': len(dd),
                             'max_abs_prob_diff': float(np.abs(a - o).max())})
pd.DataFrame(rows).to_csv('out/bootstrap-repaired-vs-original.csv', index=False)
a = rd('rall_api', 'added-edges.csv'); a = a[a.feature_fy.eq(2024)]; d = pd.read_csv('../../data/processed/doc_recovered_edges_2024.csv', dtype=str)
S, D = set(zip(a.src_corp_code, a.target)), set(zip(d.corp_code, d.target))
out.update(rall_api_2024_added=len(S), doc_pairs=len(D), overlap=len(S & D), api_only=len(S - D), doc_only=len(D - S))
json.dump(out, open('out/compare-summary.json', 'w'), indent=1); print(json.dumps(out)); print(iso.to_string())
print(pd.DataFrame(rows).round(4).to_string())
