"""Descriptive check (no labels used): firm size, market and missing financials by ownership degree,
development years FY2019-2023 of data_v3x. Degree = undirected count over graph_{fy}.csv (Counter(src+dst)).
Usage: python src/size_by_degree.py [data_dir] -> out_size/size_by_degree.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

D = Path(sys.argv[1] if len(sys.argv) > 1 else 'data_v3x')
n = pd.read_csv(D / 'nodes.csv', dtype={'corp_code': str})
e = pd.read_csv(D / 'eligible.csv', dtype={'corp_code': str})
rows = []
for fy in range(2019, 2024):
    g = pd.read_csv(D / f'graph_{fy}.csv', dtype={'src': str, 'dst': str})
    deg = pd.concat([g.src, g.dst]).value_counts()
    el = e[e.fy == fy].merge(n[n.fy == fy], on=['fy', 'corp_code'], how='left')
    el['deg'] = el.corp_code.map(deg).fillna(0).astype(int)
    rows.append(el)
d = pd.concat(rows).reset_index(drop=True)
d['grp'] = np.select([d.deg == 0, d.deg == 1], ['0', '1'], '2+')
d['assets_bn'] = np.exp(d.log_assets) / 1e9
out = {'firm_years': len(d), 'isolated_share_by_fy': d.groupby('fy').deg.apply(lambda x: round((x == 0).mean(), 3)).to_dict()}
out['by_group'] = {}
for k, x in d.groupby('grp'):
    out['by_group'][k] = {'n': int(len(x)), 'assets_median_bn_krw': round(x.assets_bn.median(), 1),
                          'assets_q25': round(x.assets_bn.quantile(.25), 1), 'assets_q75': round(x.assets_bn.quantile(.75), 1),
                          'financials_missing': round(x.log_assets.isna().mean(), 3),
                          'market_share': x.market.fillna('NA').value_counts(normalize=True).round(3).to_dict()}
d['q'] = pd.qcut(d.log_assets, 4, labels=['Q1_small', 'Q2', 'Q3', 'Q4_large'])
out['isolated_share_by_asset_quartile'] = d.groupby('q', observed=True).deg.apply(lambda x: round((x == 0).mean(), 3)).to_dict()
a, b = d[d.grp == '0'].log_assets.dropna(), d[d.grp != '0'].log_assets.dropna()
u = mannwhitneyu(b, a)
out['P_connected_larger_than_isolated'] = round(u.statistic / (len(a) * len(b)), 3)
out['mannwhitney_p'] = float(u.pvalue)
Path('out_size').mkdir(exist_ok=True)
Path('out_size/size_by_degree.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print(json.dumps(out, indent=1))
