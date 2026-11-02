"""Baseline build_panel.py logic (reference: baseline src/build_panel.py, unchanged rules), parameterised only by GRAPH.
Same inputs (sha256-identical interim files) + rebuilt financial panel. Graph variants:
  original : baseline edges
  r2024doc : baseline edges + doc-verified recovered edges for FY2024 only (data/processed/doc_recovered_edges_2024.csv)
  rall_api : all years 2021-2024, same API-only rules every year: (A) API row whose rcept_no is a correction filed after C
             is kept iff an original annual report for that FY was filed <=C (proxy; 2024 checked against documents),
             (B) investee names also resolved via verified as-filed former names (filed <=C), unique, not a current name.
Usage: build_panel.py VARIANT OUTDIR [--neighbor-asof]
AS-FILED step (reruns/asfiled-financials): graph fixed at r2024doc; late-label samples restored with
as-filed (<=label cutoff) OI from data/processed/asof_financials.csv. --neighbor-asof additionally lets
firms whose feature-year financials were late-corrected serve as NEIGHBOURS using as-of values (never as samples)."""
import collections, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from fin_panel import build as build_fin
P = Path(__file__).resolve().parents[3]; RUN = Path(__file__).resolve().parents[1]
VARIANT, OUT = sys.argv[1], Path(sys.argv[2]); NB_ASOF = '--neighbor-asof' in sys.argv; OUT.mkdir(parents=True, exist_ok=True); (OUT / 'graphs').mkdir(exist_ok=True)
read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
V = read(P / 'data/interim/invest_panel_raw.csv'); U = read(P / 'data/processed/universe_firms.csv'); O = read(P / 'data/interim/company_overview.csv'); A = read(P / 'data/interim/filings_A001.csv')
F = read(P / 'data/interim/fin_multi_raw.csv'); fil = A.drop_duplicates('rcept_no').set_index('rcept_no')
F = F[F.fs_div.eq('OFS')].copy(); F['disclosure'] = F.rcept_no.map(fil.rcept_dt)
fd = F.groupby(['corp_code', 'bsns_year']).disclosure.max().rename('financial_disclosure').reset_index().rename(columns={'bsns_year': 'fy'}); fd.fy = fd.fy.astype(int)
fp = P / 'reruns/repaired-edges/fin_panel_rebuilt.csv'
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
AS = pd.read_csv(P / 'data/processed/asof_financials.csv', dtype={'corp_code': str, 'cutoff': str, 'asof_operating_income_rcept_no': str})
AS = AS.set_index(['corp_code', 'fy', 'cutoff'])
def asof(c, fy, cut):
    k = (c, fy, cut)
    return AS.loc[k] if k in AS.index else None
restored = []
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
    nb_added = 0
    if NB_ASOF and year in (2023, 2024):   # as-of values dated <= this year's C exist only for (2023,C=20240430),(2024,C=20250430)
        for (c, fy, cut), r in AS.iterrows():
            if fy != year or cut != cutoff or c in features.index or c not in observed or not (r.asof_assets > 0): continue
            f = {'log_assets': np.log1p(r.asof_assets)}
            for name in ('liabilities', 'equity', 'capital', 'revenue', 'operating_income', 'net_income'): f[name + '_assets'] = r['asof_' + name] / r.asof_assets
            for name in ('revenue', 'operating_income'): f['signed_log_' + name] = np.sign(r['asof_' + name]) * np.log1p(abs(r['asof_' + name]))
            f['current_operating_loss'] = float(r.asof_operating_income < 0) if pd.notna(r.asof_operating_income) else np.nan
            features.loc[c] = pd.Series(f)[fields].replace([np.inf, -np.inf], np.nan); nb_added += 1
    nextp = panel[panel.fy.eq(year + 1) & panel.currencies.eq('KRW') & panel.asset_period_end.eq(str(year + 1) + '1231') & panel.operating_income.notna()].set_index('corp_code')
    candidate = sorted(set(x.index) & set(nextp.index)); known = [c for c in candidate if c in source_obs]
    eligible = [c for c in known if np.isfinite(x.loc[c, 'operating_income'])]
    label_dropped = 0
    if year in (2021, 2022):
        dr = [c for c in eligible if nextp.loc[c, 'financial_disclosure'] > '20240430']; label_dropped = len(dr)
        dropped_train += [dict(corp_code=c, feature_fy=year, label_fy=year + 1, target_disclosure=nextp.loc[c, 'financial_disclosure']) for c in dr]
        eligible = [c for c in eligible if nextp.loc[c, 'financial_disclosure'] <= '20240430' or asof(c, year + 1, '20240430') is not None and pd.notna(asof(c, year + 1, '20240430').asof_operating_income)]
    split = 'train' if year in (2021, 2022) else 'validation' if year == 2023 else 'test'
    lab_cut = '20240430' if year in (2021, 2022) else '20250430'
    for c in eligible:
        label = int(nextp.loc[c, 'operating_income'] < 0); tdisc = nextp.loc[c, 'financial_disclosure']; rest = False
        if year <= 2023 and tdisc > lab_cut:   # final label version is late -> as-filed version dated <= lab_cut
            a = asof(c, year + 1, lab_cut)
            if a is None or pd.isna(a.asof_operating_income): continue
            restored.append(dict(corp_code=c, feature_fy=year, final_label=label, asof_label=int(a.asof_operating_income < 0), final_disclosure=tdisc,
                                 asof_rcept_no=a.asof_operating_income_rcept_no, src=a.asof_operating_income_src))
            label = int(a.asof_operating_income < 0); tdisc = str(a.asof_operating_income_rcept_no)[:8]; rest = True
        ns = sorted(und[c]); available = [n for n in ns if n in features.index]
        base = {'corp_code': c, 'feature_fy': year, 'outcome_fy': year + 1, 'prediction_date': cutoff, 'split': split,
                'financial_disclosure': x.loc[c, 'financial_disclosure'], 'target_disclosure': tdisc,
                'label': label, 'restored_asof_label': rest, 'degree': len(ns), 'in_degree': len(incoming[c]), 'out_degree': len(outgoing[c]),
                'neighbor_financial_count': len(available), 'source_investment_observed': True}
        base.update(features.loc[c].to_dict())
        for name in ('degree', 'in_degree', 'out_degree'): base['log1p_' + name] = np.log1p(base[name])
        means = features.loc[available].mean() if available else pd.Series(np.nan, index=fields)
        for name in fields: base['neighbor_mean_' + name] = float(means[name])
        base['neighbor_financial_fraction'] = len(available) / len(ns) if ns else 0.
        rows_all.append(base)
    graph_stats.append({'feature_fy': year, 'observed_nodes': len(observed), 'directed_edges': len(pairs), 'added_edges': len(add),
                        'undirected_edges': sum(map(len, und.values())) // 2, 'isolated_observed': sum(not ns for ns in und.values()), 'neighbor_asof_added': nb_added})
data = pd.DataFrame(rows_all).sort_values(['feature_fy', 'corp_code']).reset_index(drop=True)
excluded = data[data.split.eq('validation') & data.target_disclosure.gt('20250430')]   # baseline finalize step: 314 late validation labels
data = data.drop(excluded.index)
pd.DataFrame(restored).to_csv(OUT / 'restored-samples.csv', index=False)
data.to_csv(OUT / 'model-panel.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(graph_stats).to_csv(OUT / 'annual-graph-stats.csv', index=False)
pd.DataFrame(dropped_train).to_csv(OUT / 'dropped-train-labels.csv', index=False)
pd.DataFrame(added_log).to_csv(OUT / 'added-edges.csv', index=False)
print(json.dumps({'variant': VARIANT, 'rows': len(data), 'excluded_validation': len(excluded), 'split': data.split.value_counts().to_dict(), 'graphs': graph_stats}, default=int))
