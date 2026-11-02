"""Paired comparison asfiled vs r2024doc, decision-rule bootstrap, sub-analyses (a)(b), FP/FN for newly connected firms."""
import json, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as ROC, brier_score_loss as BR
rd = lambda p: pd.read_csv(p, dtype={'corp_code': str})
RE = '../repaired-edges/out/'
pan = {'original': rd(RE + 'original/model-panel.csv'), 'r2024doc': rd(RE + 'r2024doc/model-panel.csv'), 'main': rd('out/main/model-panel.csv'), 'nbasof': rd('out/nbasof/model-panel.csv')}
pred = {'original': rd(RE + 'original/predictions.csv'), 'r2024doc': rd(RE + 'r2024doc/predictions.csv'), 'main': rd('out/main/predictions.csv'), 'nbasof': rd('out/nbasof/predictions.csv')}
test = pan['r2024doc'][pan['r2024doc'].split.eq('test')].reset_index(drop=True)
for k in pan: assert sorted(pan[k].query("split=='test'").corp_code) == sorted(test.corp_code)
y = test.label.to_numpy(); C = test.corp_code
FAM = ('logistic', 'hist_gradient_boosting'); CON = ('financial', 'financial_degree', 'financial_degree_neighbors')
def P(v, fam, cond, codes=C, col='probability'):
    q = pred[v]; q = q[q.split.eq('test') & q.family.eq(fam) & q.features.eq(cond)].set_index('corp_code'); return q.loc[codes, col].to_numpy()
def boot(yv, a, b, f=AP):
    rng = np.random.default_rng(20261102); d = []
    for _ in range(500):
        ix = rng.integers(0, len(yv), len(yv))
        if len(np.unique(yv[ix])) == 2: d.append(f(yv[ix], a[ix]) - f(yv[ix], b[ix]))
    return f(yv, a) - f(yv, b), np.percentile(d, 2.5), np.percentile(d, 97.5)
res = {}
# 1. paired vs r2024doc
rows = []
for v in ('main', 'nbasof'):
    for fam in FAM:
        for cond in CON:
            a, b = P(v, fam, cond), P('r2024doc', fam, cond)
            d, lo, hi = boot(y, a, b); dr, lr, hr = boot(y, a, b, ROC)
            rows.append(dict(variant=v, family=fam, features=cond, AP=AP(y, a), AP_r2024doc=AP(y, b), dAP=d, dAP_lo=lo, dAP_hi=hi,
                             ROC=ROC(y, a), ROC_r2024doc=ROC(y, b), dROC=dr, dROC_lo=lr, dROC_hi=hr, Brier=BR(y, a), Brier_r2024doc=BR(y, b)))
pd.DataFrame(rows).to_csv('out/paired-vs-r2024doc.csv', index=False)
# neighbours (and missind) minus financial within variant, plus subsets
rows = []
for v in ('r2024doc', 'main', 'nbasof'):
    t = pan[v].query("split=='test'").set_index('corp_code').loc[C]
    allfin = (t.degree.gt(0) & t.neighbor_financial_count.eq(t.degree)).to_numpy()
    for sub, m in (('all_test', np.ones(len(C), bool)), ('connected_all_neighbors_have_financials', allfin)):
        for fam in FAM:
            for cond in ('financial_degree_neighbors', 'financial_degree_neighbors_missind'):
                if v == 'r2024doc' and cond.endswith('missind'): continue
                for ref in ('financial',) + (('financial_degree_neighbors',) if cond.endswith('missind') else ()):
                    a, b = P(v, fam, cond)[m], P(v, fam, ref)[m]; ys = y[m]
                    d, lo, hi = boot(ys, a, b); dr, lr, hr = boot(ys, a, b, ROC)
                    rows.append(dict(variant=v, subset=sub, n=int(m.sum()), pos=int(ys.sum()), family=fam, comparison=f'{cond} minus {ref}',
                                     AP_model=AP(ys, a), AP_ref=AP(ys, b), dAP=d, lo=lo, hi=hi, ROC_model=ROC(ys, a), ROC_ref=ROC(ys, b), dROC=dr, ROC_lo=lr, ROC_hi=hr))
W = pd.DataFrame(rows); W.to_csv('out/neighbors-vs-financial.csv', index=False)
# 3. FP/FN
o = pan['original'].query("split=='test'").set_index('corp_code').loc[C]; t = pan['main'].query("split=='test'").set_index('corp_code').loc[C]
newly = (o.degree.eq(0) & t.degree.gt(0)).to_numpy()
grp = np.where(t.degree.eq(0), 'isolated', np.where(t.neighbor_financial_count.eq(t.degree), 'connected_all_nbr_fin', 'connected_missing_nbr_fin'))
rows = []
for gname, mask in [('newly_connected', newly)] + [(g, grp == g) for g in ('isolated', 'connected_all_nbr_fin', 'connected_missing_nbr_fin')] + [('newly_connected&all_nbr_fin', newly & (grp == 'connected_all_nbr_fin')), ('newly_connected&missing_nbr_fin', newly & (grp == 'connected_missing_nbr_fin'))]:
    for v in ('original', 'r2024doc', 'main', 'nbasof'):
        for fam in FAM:
            for cond in CON:
                p = P(v, fam, cond)[mask]; th = P(v, fam, cond, col='threshold')[mask]; ys = y[mask]; pr = p >= th
                fp, fn = int((pr & (ys == 0)).sum()), int((~pr & (ys == 1)).sum())
                rows.append(dict(group=gname, n=int(mask.sum()), pos=int(ys.sum()), variant=v, family=fam, features=cond, FP=fp, FN=fn,
                                 FP_rate=fp / max((ys == 0).sum(), 1), FN_rate=fn / max(ys.sum(), 1)))
pd.DataFrame(rows).to_csv('out/fp-fn.csv', index=False)
res.update(newly_connected=int(newly.sum()), groups={g: int((grp == g).sum()) for g in set(grp)})
for v in pan:
    q = pan[v].query("split=='test'"); res[f'{v}_test_connected_no_nbr_fin'] = int((q.degree.gt(0) & q.neighbor_financial_count.eq(0)).sum())
    res[f'{v}_test_connected_some_nbr_missing'] = int((q.degree.gt(0) & q.neighbor_financial_count.lt(q.degree)).sum())
# test-side neighbour features: does main change any vs r2024doc?
nc = [c for c in test.columns if c.startswith('neighbor_')]
a = pan['main'].query("split=='test'").set_index('corp_code').loc[C, nc]; b = test.set_index('corp_code').loc[C, nc]
res['main_vs_r2024doc_test_neighbor_feature_cells_changed'] = int((~((a == b) | (a.isna() & b.isna()))).to_numpy().sum())
a = pan['nbasof'].query("split=='test'").set_index('corp_code').loc[C, nc]
res['nbasof_vs_r2024doc_test_rows_neighbor_features_changed'] = int((~((a == b) | (a.isna() & b.isna()))).any(axis=1).sum())
json.dump(res, open('out/analysis-summary.json', 'w'), indent=1); print(json.dumps(res, indent=1))
