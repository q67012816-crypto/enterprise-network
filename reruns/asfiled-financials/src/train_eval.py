"""Train the six fixed baseline models on one graph-variant panel; thresholds = validation max-F1 on the
2,071 as-of-corrected validation rows (baseline finalize step); test once. Mirrors baseline
src/train_models.py + src/finalize_enterprise_baseline.py (no tuning). Usage: train_eval.py OUTDIR"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve
OUT = Path(sys.argv[1]); RUN = Path(__file__).resolve().parents[1]
d = pd.read_csv(OUT / 'model-panel.csv', dtype={'corp_code': str, 'financial_disclosure': str, 'target_disclosure': str, 'prediction_date': str})
schema = json.loads((RUN.parent / 'repaired-edges/feature-schema.json').read_text())
schema['financial_degree_neighbors_missind'] = schema['financial_degree_neighbors'] + ['any_neighbor_missing_financials']   # sub-analysis (b): the one added indicator
d['any_neighbor_missing_financials'] = (d.degree.gt(0) & d.neighbor_financial_count.lt(d.degree)).astype(float)
train, val, test = (d[d.split.eq(s)] for s in ('train', 'validation', 'test'))
assert train.target_disclosure.le('20240430').all() and val.target_disclosure.le('20250430').all()
rows, groups, preds = [], [], []
def metric(y, p, t):
    y = np.asarray(y); pred = p >= t
    tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum()); fn = int(((y == 1) & ~pred).sum()); tn = int(((y == 0) & ~pred).sum())
    return {'n': len(y), 'positive': int(y.sum()), 'AP': float(average_precision_score(y, p)),
            'ROC_AUC': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None, 'Brier': float(brier_score_loss(y, p)),
            'accuracy': (tp + tn) / len(y), 'recall': tp / (tp + fn) if tp + fn else None, 'precision': tp / (tp + fp) if tp + fp else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0., 'threshold': float(t), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
def record(frame, p, t, family, cond):
    split = frame.split.iloc[0]; rows.append({'split': split, 'family': family, 'features': cond, **metric(frame.label, p, t)})
    for name, mask in [('isolated', frame.degree.eq(0)), ('connected', frame.degree.gt(0)), ('degree1', frame.degree.eq(1)),
                       ('degree2_3', frame.degree.between(2, 3)), ('degree4plus', frame.degree.ge(4))]:
        if mask.any(): groups.append({'split': split, 'family': family, 'features': cond, 'group': name, **metric(frame.loc[mask, 'label'], p[mask.to_numpy()], t)})
    preds.append(pd.DataFrame({'corp_code': frame.corp_code, 'feature_fy': frame.feature_fy, 'split': split, 'degree': frame.degree,
                               'label': frame.label, 'family': family, 'features': cond, 'probability': p, 'threshold': t}))
for family in ('logistic', 'hist_gradient_boosting'):
    for cond, cols in schema.items():
        lo, hi = train[cols].quantile(.005), train[cols].quantile(.995)
        clip = lambda f: f[cols].clip(lo, hi, axis=1)
        if family == 'logistic': m = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegression(C=1, solver='lbfgs', max_iter=2000, random_state=20261102))
        else: m = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), HistGradientBoostingClassifier(learning_rate=.05, max_iter=200, max_leaf_nodes=8, min_samples_leaf=30, l2_regularization=1, early_stopping=False, random_state=20261102))
        m.fit(clip(train), train.label); pv = m.predict_proba(clip(val))[:, 1]
        pr, rc, th = precision_recall_curve(val.label, pv); f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-15)
        t = float(th[np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]])
        record(val, pv, t, family, cond); record(test, m.predict_proba(clip(test))[:, 1], t, family, cond)
P = pd.concat(preds, ignore_index=True); P.to_csv(OUT / 'predictions.csv', index=False)
pd.DataFrame(rows).to_csv(OUT / 'metrics.csv', index=False); pd.DataFrame(groups).to_csv(OUT / 'group-metrics.csv', index=False)
boot = []; rng = np.random.default_rng(20261102); y = test.label.to_numpy()   # identical scheme/seed to baseline
for family in ('logistic', 'hist_gradient_boosting'):
    part = P[P.split.eq('test') & P.family.eq(family)]
    s = {n: part[part.features.eq(n)].set_index('corp_code').loc[test.corp_code, 'probability'].to_numpy() for n in schema}
    for ref in ('financial', 'financial_degree'):
        diffs = {'AP': [], 'ROC_AUC': []}
        for _ in range(500):
            ix = rng.integers(0, len(y), len(y)); yy = y[ix]
            if len(np.unique(yy)) < 2: continue
            for n, f in [('AP', average_precision_score), ('ROC_AUC', roc_auc_score)]: diffs[n].append(float(f(yy, s['financial_degree_neighbors'][ix]) - f(yy, s[ref][ix])))
        for n, v in diffs.items():
            f = average_precision_score if n == 'AP' else roc_auc_score
            boot.append({'family': family, 'comparison': 'financial_degree_neighbors minus ' + ref, 'metric': n,
                         'difference': float(f(y, s['financial_degree_neighbors']) - f(y, s[ref])), 'ci95_low': float(np.percentile(v, 2.5)),
                         'ci95_high': float(np.percentile(v, 97.5)), 'bootstrap_replicates': len(v)})
pd.DataFrame(boot).to_csv(OUT / 'paired-bootstrap.csv', index=False)
print(pd.DataFrame(rows).query("split=='test'")[['family', 'features', 'AP', 'ROC_AUC', 'Brier']].to_string())
