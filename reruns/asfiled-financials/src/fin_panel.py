"""Rebuild the baseline's financial-OFS panel (corp_code, fy, accounts, currencies, asset_period_end) from
data/interim/fin_multi_raw.csv (sha256 identical to baseline dependency). The original audit-panel file is not
in the baseline zip; this reconstruction is validated against model-panel.csv features (see check_fin_panel)."""
import re, numpy as np, pandas as pd
from pathlib import Path
P = Path(__file__).resolve().parents[3]
MAP = {'assets': ['자산총계'], 'liabilities': ['부채총계'], 'equity': ['자본총계'], 'capital': ['자본금'],
       'revenue': ['매출액'], 'operating_income': ['영업이익', '영업이익(손실)'], 'net_income': ['당기순이익(손실)']}
def build():
    F = pd.read_csv(P / 'data/interim/fin_multi_raw.csv', dtype=str, keep_default_na=False)
    F = F[F.fs_div.eq('OFS')].copy()
    F['v'] = pd.to_numeric(F.thstrm_amount.str.replace(',', '', regex=False), errors='coerce')
    rows = []
    for (c, y), g in F.groupby(['corp_code', 'bsns_year'], sort=False):
        r = {'corp_code': c, 'fy': int(y), 'currencies': ','.join(sorted(set(g.currency) - {''}))}
        for k, names in MAP.items():
            h = g[g.account_nm.isin(names)]
            if k == 'net_income': h = h[h.sj_div.eq('IS')] if h.sj_div.eq('IS').any() else h
            u = h.v.dropna().unique()
            r[k] = u[0] if len(u) == 1 else np.nan   # conflicting duplicate values -> missing (matches baseline panel, e.g. 00161444)
        a = g[g.account_nm.eq('자산총계')].thstrm_dt
        m = re.search(r'(\d{4})\.(\d{2})\.(\d{2})', a.iloc[0]) if len(a) else None
        r['asset_period_end'] = ''.join(m.groups()) if m else ''
        rows.append(r)
    return pd.DataFrame(rows)
if __name__ == '__main__':
    d = build(); d.to_csv(Path(__file__).resolve().parents[1] / 'fin_panel_rebuilt.csv', index=False); print(len(d))
