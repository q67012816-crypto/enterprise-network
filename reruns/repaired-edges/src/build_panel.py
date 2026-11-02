"""Baseline build_panel.py logic (reference: baseline src/build_panel.py, unchanged rules), parameterised only by GRAPH.
Same inputs (sha256-identical interim files) + rebuilt financial panel. Graph variants:
  original : baseline edges
  r2024doc : baseline edges + doc-verified recovered edges for FY2024 only (data/processed/doc_recovered_edges_2024.csv)
  rall_api : all years 2021-2024, same API-only rules every year: (A) API row whose rcept_no is a correction filed after C
             is kept iff an original annual report for that FY was filed <=C (proxy; 2024 checked against documents),
             (B) investee names also resolved via verified as-filed former names (filed <=C), unique, not a current name.
Usage: build_panel.py VARIANT OUTDIR"""
import collections, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from fin_panel import build as build_fin
P = Path(__file__).resolve().parents[3]; RUN = Path(__file__).resolve().parents[1]
VARIANT, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True); (OUT / 'graphs').mkdir(exist_ok=True)
read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
V = read(P / 'data/interim/invest_panel_raw.csv'); U = read(P / 'data/processed/universe_firms.csv'); O = read(P / 'data/interim/company_overview.csv'); A = read(P / 'data/interim/filings_A001.csv')
F = read(P / 'data/interim/fin_multi_raw.csv'); fil = A.drop_duplicates('rcept_no').set_index('rcept_no')
F = F[F.fs_div.eq('OFS')].copy(); F['disclosure'] = F.rcept_no.map(fil.rcept_dt)
fd = F.groupby(['corp_code', 'bsns_year']).disclosure.max().rename('financial_disclosure').reset_index().rename(columns={'bsns_year': 'fy'}); fd.fy = fd.fy.astype(int)
fp = RUN / 'fin_panel_rebuilt.csv'
panel = pd.read_csv(fp, dtype={'corp_code': str, 'asset_period_end': str, 'currencies': str}, keep_default_na=False, na_values=[''])
panel['currencies'] = panel.currencies.fillna('')
panel = panel.merge(fd, on=['corp_code', 'fy'], validate='one_to_one')
fields = ['log_assets', 'liabilities_assets', 'equity_assets', 'capital_assets', 'revenue_assets', 'operating_income_assets', 'net_income_assets', 'signed_log_revenue', 'signed_log_operating_income', 'current_operating_loss']
suffix = re.compile(r'\(주\)|㈜|주식회사|\(유\)|유한회사|co\.?,?\s*ltd\.?|inc\.?|corp\.?|\s+', re.I)
norm = lambda s: suffix.sub('', s or '').replace('（', '(').replace('）', ')').strip().lower()
codes = set(U.corp_code); names = {k: collections.defaultdict(set) for k in ('exact', 'norm', 'stock')}
for row in O.to_dict('records'):
    if row['corp_code'] not in codes: continue
    for key, value in [('exact', row['corp_name'].strip()), ('norm', norm(row['corp_name'])), ('stock', norm(row['stock_name']))]:
        if value: names[key][value].add(row['corp_code'])
def resolve(value):
    for key, name in [('exact', value.strip()), ('norm', norm(value)), ('stock', norm(value))]:
        c = names[key].get(name, set())
        if len(c) == 1: return next(iter(c)), key
        if len(c) > 1: return '', 'ambiguous'
    return '', 'unmatched'
resolved = {s: resolve(s) for s in V.inv_prm.unique()}
V['target'] = V.inv_prm.map(lambda s: resolved[s][0]); V['match_rule'] = V.inv_prm.map(lambda s: resolved[s][1])
V['disclosure'] = V.rcept_no.map(fil.rcept_dt); V['report_name'] = V.rcept_no.map(fil.report_nm)
def numeric(col): return pd.to_numeric(col.str.replace(',', '', regex=False), errors='coerce')
V['pct'] = numeric(V.trmend_blce_qota_rt); V['quantity'] = numeric(V.trmend_blce_qy); V['book'] = numeric(V.trmend_blce_acntbk_amount)
V['invalid_pct'] = V.pct.notna() & (~V.pct.between(0, 100)); V['held'] = (V.pct.gt(0) & V.pct.le(100)) | V.quantity.gt(0) | V.book.gt(0)
annual = A[A.report_nm.str.match(r'^사업보고서\s*\(') & ~A.report_nm.str.contains('연장신고')].copy()
annual['fy'] = annual.report_nm.str.extract(r'사업보고서\s*\((\d{4})\.')[0]
if VARIANT == 'rall_api':
    sys.path.insert(0, str(P / 'src')); from isolated_audit import verified_former
    orig_first = annual[~annual.report_nm.str.contains(r'\[기재정정\]|\[첨부정정\]|정정', regex=True)].groupby(['corp_code', 'fy']).rcept_dt.min()
rows_all = []; flows = []; graph_stats = []; dropped_train = []; added_log = []
for year in (2021, 2022, 2023, 2024):
    cutoff = str(year + 1) + '0430'
    observed = set(annual.loc[annual.fy.eq(str(year)) & annual.rcept_dt.le(cutoff), 'corp_code']) & codes
    sub = V[V.bsns_year.eq(str(year))].copy()
    source_obs = set(sub.loc[sub.api_status.eq('000') & sub.disclosure.notna() & sub.disclosure.le(cutoff), 'src_corp_code'])
    keep = sub.api_status.eq('000') & sub.disclosure.notna() & sub.disclosure.le(cutoff) & sub.target.ne('') & ~sub.invalid_pct & sub.held
    keep &= sub.src_corp_code.isin(observed) & sub.target.isin(observed) & sub.src_corp_code.ne(sub.target)
    es = sub[keep].sort_values('disclosure').drop_duplicates(['src_corp_code', 'target'], keep='last')
    pairs = set(zip(es.src_corp_code, es.target)); add = []
    if VARIANT == 'r2024doc' and year == 2024:
        N = read(P / 'data/processed/doc_recovered_edges_2024.csv')
        for s, t, r in zip(N.corp_code, N.target, N.reason):
            if (s, t) not in pairs and s != t and s in observed and t in observed: pairs.add((s, t)); add.append((s, t, r))
    if VARIANT == 'rall_api':
        former, _ = verified_former(codes, cutoff); cur = set(names['norm'])
        def res2(v):
            t, k = resolved[v]
            if t or k == 'ambiguous': return t, k
            c = former.get(norm(v), set())
            return (next(iter(c)), 'former_name') if len(c) == 1 and norm(v) not in cur else ('', k)
        tg = sub.inv_prm.map(lambda v: res2(v)); sub['t2'] = tg.str[0]; sub['r2'] = tg.str[1]
        orig_ok = [orig_first.get((s, str(year)), '99999999') <= cutoff for s in sub.src_corp_code]
        timing_ok = sub.disclosure.notna() & (sub.disclosure.le(cutoff) | pd.Series(orig_ok, index=sub.index))
        k2 = sub.api_status.eq('000') & timing_ok & sub.t2.ne('') & ~sub.invalid_pct & sub.held
        k2 &= sub.src_corp_code.isin(observed) & sub.t2.isin(observed) & sub.src_corp_code.ne(sub.t2)
        for r in sub[k2].to_dict('records'):
            p = (r['src_corp_code'], r['t2'])
            if p not in pairs:
                pairs.add(p); add.append((*p, 'B_former_name' if r['r2'] == 'former_name' else 'A_correction_after_C_original_by_C'))
    added_log += [dict(feature_fy=year, src_corp_code=s, target=t, reason=r) for s, t, r in add]
    pd.DataFrame(sorted(pairs), columns=['src_corp_code', 'target']).to_csv(OUT / f'graphs/edges-{year}-asof-{cutoff}.csv', index=False)
    und = {c: set() for c in observed}; incoming = {c: set() for c in observed}; outgoing = {c: set() for c in observed}
    for s, t in pairs: und[s].add(t); und[t].add(s); outgoing[s].add(t); incoming[t].add(s)
    x = panel[panel.fy.eq(year) & panel.currencies.eq('KRW') & panel.assets.gt(0) & panel.asset_period_end.eq(str(year) + '1231') & panel.financial_disclosure.le(cutoff)].copy()
    x = x[x.corp_code.isin(observed)].set_index('corp_code')
    features = pd.DataFrame(index=x.index)
    features['log_assets'] = np.log1p(x.assets)
    for name in ('liabilities', 'equity', 'capital', 'revenue', 'operating_income', 'net_income'): features[name + '_assets'] = x[name] / x.assets
    for name in ('revenue', 'operating_income'): features['signed_log_' + name] = np.sign(x[name]) * np.log1p(x[name].abs())
    features['current_operating_loss'] = x.operating_income.lt(0).where(x.operating_income.notna()).astype(float)
    features = features[fields].replace([np.inf, -np.inf], np.nan)
    nextp = panel[panel.fy.eq(year + 1) & panel.currencies.eq('KRW') & panel.asset_period_end.eq(str(year + 1) + '1231') & panel.operating_income.notna()].set_index('corp_code')
    candidate = sorted(set(x.index) & set(nextp.index)); known = [c for c in candidate if c in source_obs]
    eligible = [c for c in known if np.isfinite(x.loc[c, 'operating_income'])]
    label_dropped = 0
    if year in (2021, 2022):
        dr = [c for c in eligible if nextp.loc[c, 'financial_disclosure'] > '20240430']; label_dropped = len(dr)
        dropped_train += [dict(corp_code=c, feature_fy=year, label_fy=year + 1, target_disclosure=nextp.loc[c, 'financial_disclosure']) for c in dr]
        eligible = [c for c in eligible if nextp.loc[c, 'financial_disclosure'] <= '20240430']
    split = 'train' if year in (2021, 2022) else 'validation' if year == 2023 else 'test'
    for c in eligible:
        ns = sorted(und[c]); available = [n for n in ns if n in features.index]
        base = {'corp_code': c, 'feature_fy': year, 'outcome_fy': year + 1, 'prediction_date': cutoff, 'split': split,
                'financial_disclosure': x.loc[c, 'financial_disclosure'], 'target_disclosure': nextp.loc[c, 'financial_disclosure'],
                'label': int(nextp.loc[c, 'operating_income'] < 0), 'degree': len(ns), 'in_degree': len(incoming[c]), 'out_degree': len(outgoing[c]),
                'neighbor_financial_count': len(available), 'source_investment_observed': True}
        base.update(features.loc[c].to_dict())
        for name in ('degree', 'in_degree', 'out_degree'): base['log1p_' + name] = np.log1p(base[name])
        means = features.loc[available].mean() if available else pd.Series(np.nan, index=fields)
        for name in fields: base['neighbor_mean_' + name] = float(means[name])
        base['neighbor_financial_fraction'] = len(available) / len(ns) if ns else 0.
        rows_all.append(base)
    graph_stats.append({'feature_fy': year, 'observed_nodes': len(observed), 'directed_edges': len(pairs), 'added_edges': len(add),
                        'undirected_edges': sum(map(len, und.values())) // 2, 'isolated_observed': sum(not ns for ns in und.values())})
data = pd.DataFrame(rows_all).sort_values(['feature_fy', 'corp_code']).reset_index(drop=True)
excluded = data[data.split.eq('validation') & data.target_disclosure.gt('20250430')]   # baseline finalize step: 314 late validation labels
data = data.drop(excluded.index)
data.to_csv(OUT / 'model-panel.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(graph_stats).to_csv(OUT / 'annual-graph-stats.csv', index=False)
pd.DataFrame(dropped_train).to_csv(OUT / 'dropped-train-labels.csv', index=False)
pd.DataFrame(added_log).to_csv(OUT / 'added-edges.csv', index=False)
print(json.dumps({'variant': VARIANT, 'rows': len(data), 'excluded_validation': len(excluded), 'split': data.split.value_counts().to_dict(), 'graphs': graph_stats}, default=int))
