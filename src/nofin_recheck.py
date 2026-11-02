"""STEP2b: 53 connected test firms whose baseline neighbours had no usable financials. For each neighbour use the
ORIGINAL FY2024 사업보고서 filed <=C (data/raw/versions, parsed in doc_financials.csv); a neighbour counts as having
as-of financials if total assets parsed from that as-filed version. Writes data/processed/nofin_recheck.csv."""
import json, pandas as pd
from pathlib import Path
from fetch_docs import originals
R = Path(__file__).resolve().parents[1]; B = Path('/workspace/baseline-from-jack/enterprise-baseline')
rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
P = rd(B / 'model-panel.csv'); T = P[P.split.eq('test')]; E = rd(B / 'graphs/edges-2024-asof-20250430.csv')
nf = T[T.degree.ne('0') & T.neighbor_financial_count.eq('0')].corp_code
o = originals(); D = rd(R / 'data/processed/doc_financials.csv').set_index('rcept_no')
O = rd(R / 'data/interim/company_overview.csv').set_index('corp_code')
rows = []
for k in nf:
    nb = set(E[E.src_corp_code.eq(k)].target) | set(E[E.target.eq(k)].src_corp_code)
    for n in sorted(nb):
        rc = o.get(n, ''); d = D.loc[rc] if rc in D.index else None
        st = 'no_original_by_C' if not rc else 'not_downloaded' if d is None else d.doc_status
        a = (d.assets or d.get('xbrl_assets', '')) if d is not None else ''
        src = '' if not a else 'document' if d.assets else 'xbrl'
        rows.append(dict(corp_code=k, neighbor=n, neighbor_name=O.corp_name.get(n, ''), induty=O.induty_code.get(n, ''),
                         original_rcept_no=rc, doc_status=st, asof_assets=a, assets_src=src, asof_operating_income=(d.operating_income or d.get('xbrl_operating_income', '')) if d is not None else '',
                         parse_flags=d.parse_flags if d is not None else ''))
X = pd.DataFrame(rows); X.to_csv(R / 'data/processed/nofin_recheck.csv', index=False)
f = X.assign(h=X.asof_assets.ne('')).groupby('corp_code').h.any()
s = dict(firms=len(nf), neighbor_links=len(X), neighbors_with_asof_assets=int(X.asof_assets.ne('').sum()),
         firms_with_any_neighbor_asof_fin=int(f.sum()), neighbor_status=X.doc_status.value_counts().to_dict(),
         neighbors_no_assets_flags=X[X.asof_assets.eq('')].parse_flags.value_counts().head(8).to_dict())
json.dump(s, open(R / 'reports/nofin_recheck.json', 'w'), indent=1, ensure_ascii=False); print(json.dumps(s, ensure_ascii=False, indent=1))
