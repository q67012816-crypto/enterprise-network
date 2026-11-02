"""STEP1(b): as-of table for late-label samples. For each target (corp, fy): the version used at prediction cut-off
C (validation FY2024: 20250430; training FY2022/2023 labels: 20240430) = latest A001 version with rcept_dt <= C whose
document parsed the account. Compares as-of vs final (fnlttMultiAcnt = latest version) and the parser vs final.
Outputs data/processed/asof_financials.csv, reports/asof_recovery.json."""
import json, numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]; B = Path('/workspace/baseline-from-jack/enterprise-baseline')
ACC = {'assets': '자산총계', 'liabilities': '부채총계', 'equity': '자본총계', 'capital': '자본금', 'revenue': '매출액',
       'operating_income': ['영업이익', '영업이익(손실)'], 'net_income': '당기순이익(손실)'}
K = list(ACC)

def final_values():
    F = pd.read_csv(R / 'data/interim/fin_multi_raw.csv', dtype=str); F = F[F.fs_div.eq('OFS')]
    out = {}
    for k, names in ACC.items():
        names = names if isinstance(names, list) else [names]
        s = F[F.account_nm.isin(names)]
        if k == 'net_income': s = s[s.sj_div.eq('IS')] if s.sj_div.eq('IS').any() else s
        s = s.drop_duplicates(['corp_code', 'bsns_year'])
        out[k] = pd.to_numeric(s.set_index(['corp_code', 'bsns_year']).thstrm_amount.str.replace(',', ''), errors='coerce')
    return pd.DataFrame(out)

def main():
    V = pd.read_csv(R / 'data/processed/versions_index.csv', dtype=str)
    D = pd.read_csv(R / 'data/processed/doc_financials.csv', dtype={'rcept_no': str, 'corp_code': str, 'fy': str, 'rcept_dt': str})
    for a in K:  # as-filed value: document table cell; fallback XBRL instance of the same rcept_no (flagged)
        D[a + '_src'] = np.where(D[a].notna(), 'document', np.where(D.get('xbrl_' + a, pd.Series(np.nan, index=D.index)).notna(), 'xbrl', ''))
        D[a] = D[a].fillna(D.get('xbrl_' + a, pd.Series(np.nan, index=D.index)))
    D['doc_status'] = np.where(D.doc_status.eq('ok') | D.xbrl_sha1.str.len().eq(40), 'ok', D.doc_status)
    V = V.merge(D[['rcept_no', 'doc_status', 'xbrl_sha1'] + K + [a + '_src' for a in K]], on='rcept_no', how='left')
    FV = final_values(); rows = []
    for (k, fy, grp), g in V.groupby(['corp_code', 'fy', 'group']):
        C = '20250430' if grp == 'validation_314' else '20240430'
        g = g.sort_values('rcept_no'); fin = FV.loc[(k, fy)] if (k, fy) in FV.index else None
        rec = dict(corp_code=k, fy=fy, group=grp, cutoff=C, n_versions=int(g.rcept_no.notna().sum()),
                   n_versions_by_C=int((g.rcept_dt <= C).sum()), first_rcept_dt=g.rcept_dt.min(), last_rcept_dt=g.rcept_dt.max(),
                   docs_downloaded=int(g.doc_status.eq('ok').sum()),
                   xbrl_distinct=g.xbrl_sha1.dropna()[~g.xbrl_sha1.dropna().str.startswith('not_zip')].nunique())
        pre = g[(g.rcept_dt <= C) & g.doc_status.eq('ok')]; last = g[g.doc_status.eq('ok')].tail(1)
        for a in K:
            p = pre[a].dropna(); rec[f'asof_{a}'] = p.iloc[-1] if len(p) else np.nan
            rec[f'asof_{a}_rcept_no'] = pre.loc[p.index[-1], 'rcept_no'] if len(p) else ''
            rec[f'asof_{a}_src'] = pre.loc[p.index[-1], a + '_src'] if len(p) else ''
            rec[f'final_{a}'] = fin[a] if fin is not None else np.nan
            rec[f'lastdoc_{a}'] = last[a].iloc[0] if len(last) else np.nan
        rows.append(rec)
    T = pd.DataFrame(rows); T.to_csv(R / 'data/processed/asof_financials.csv', index=False)
    S = {}
    for grp, t in T.groupby('group'):
        s = {'targets': len(t), 'with_version_by_C': int((t.n_versions_by_C > 0).sum()),
             'asof_core_recovered(assets & operating_income)': int((t.asof_assets.notna() & t.asof_operating_income.notna()).sum()),
             'asof_label_recovered(operating_income)': int(t.asof_operating_income.notna().sum())}
        for a in K:
            m = t[f'asof_{a}'].notna() & t[f'final_{a}'].notna()
            pc = t[f'lastdoc_{a}'].notna() & t[f'final_{a}'].notna()
            s[a] = {'compared': int(m.sum()), 'asof_differs_from_final': int((~np.isclose(t.loc[m, f'asof_{a}'], t.loc[m, f'final_{a}'], rtol=1e-9, atol=0.5)).sum()),
                    'parser_check_lastdoc_vs_final_n': int(pc.sum()),
                    'parser_check_agree': int(np.isclose(t.loc[pc, f'lastdoc_{a}'], t.loc[pc, f'final_{a}'], rtol=1e-9, atol=0.5).sum())}
        m = t.asof_operating_income.notna() & t.final_operating_income.notna()
        s['label_flip_operating_loss'] = int(((t.loc[m, 'asof_operating_income'] < 0) != (t.loc[m, 'final_operating_income'] < 0)).sum())
        s['label_compared'] = int(m.sum())
        s['xbrl_versions_distinct_gt1'] = int((t.xbrl_distinct > 1).sum())
        S[grp] = s
    json.dump(S, open(R / 'reports/asof_recovery.json', 'w'), indent=1, ensure_ascii=False); print(json.dumps(S, indent=1, ensure_ascii=False))

if __name__ == '__main__': main()
