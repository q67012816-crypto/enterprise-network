"""STEP2: why are baseline test (FY2024, C=20250430) firms isolated?  Reproduces baseline resolution rules
(exact / suffix-normalised / stock-name, unique only) on data/interim/invest_panel_raw.csv, checks the isolated set,
then classifies each isolated firm. Writes data/processed/isolated_audit.csv and reports/isolated_audit.json.
Name-history rule (ii) uses data/processed/name_history.csv if present (effective-dated names from filings)."""
import collections, json, re, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]; B = Path('/workspace/baseline-from-jack/enterprise-baseline')
read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
suffix = re.compile(r'\(주\)|㈜|주식회사|\(유\)|유한회사|co\.?,?\s*ltd\.?|inc\.?|corp\.?|\s+', re.I)
norm = lambda s: suffix.sub('', s or '').replace('（', '(').replace('）', ')').strip().lower()
PLACEHOLDER = {'', '-', '합계', '계', '소계', '총계', '해당사항없음', '해당사항 없음', '없음'}
YEAR, C = '2024', '20250430'

def build_names(O, codes, extra=None):
    names = {k: collections.defaultdict(set) for k in ('exact', 'norm', 'stock')}
    for r in O.to_dict('records'):
        if r['corp_code'] not in codes: continue
        for k, v in [('exact', r['corp_name'].strip()), ('norm', norm(r['corp_name'])), ('stock', norm(r['stock_name']))]:
            if v: names[k][v].add(r['corp_code'])
    return names

def verified_former(codes, C=C):
    """Former names from data/processed/name_history.csv, effective-dated & chained to the firm's OWN filed name:
    a '상호의 변경' (before, after) row of corp X is accepted only if `after` equals (normalised) a COMPANY-NAME X filed
    itself, or an already-accepted name (iterated). Rows from filings after C are ignored. Rejects subsidiary rename tables."""
    p = R / 'data/processed/name_history.csv'; out = collections.defaultdict(set)
    if not p.exists(): return out, False
    H = read(p); H = H[H.corp_code.isin(codes) & H.source_verified.eq('Y') & H.rcept_no.str[:8].le(C)]
    own = {k: set(map(norm, g.name)) for k, g in H[H.source.eq('COMPANY-NAME as filed')].groupby('corp_code')}
    kb = ['corp_code', 'rcept_no', 'change_date']
    P = H[H.source.str.endswith('before')][kb + ['name']].merge(H[H.source.str.endswith('after')][kb + ['name']], on=kb, suffixes=('_b', '_a'))
    pairs = list(zip(P.corp_code, P.name_b, P.name_a))
    acc = {k: set(v) for k, v in own.items()}
    for _ in range(5):
        for k, b, a in pairs:
            if norm(a) in acc.get(k, set()) and norm(b): acc.setdefault(k, set()).add(norm(b))
    for k, v in acc.items():
        for n in v: out[n].add(k)
    return out, True

def resolve(names, v):
    for k, n in [('exact', v.strip()), ('norm', norm(v)), ('stock', norm(v))]:
        c = names[k].get(n, set())
        if len(c) == 1: return next(iter(c)), k
        if len(c) > 1: return '', 'ambiguous'
    return '', 'unmatched'

def main():
    U = read(R / 'data/processed/universe_firms.csv'); O = read(R / 'data/interim/company_overview.csv')
    A = read(R / 'data/interim/filings_A001.csv'); V = read(R / 'data/interim/invest_panel_raw.csv')
    fil = A.drop_duplicates('rcept_no').set_index('rcept_no'); codes = set(U.corp_code)
    names = build_names(O, codes)
    V = V[V.bsns_year.eq(YEAR)].copy()
    res = {s: resolve(names, s) for s in V.inv_prm.unique()}
    V['target'] = V.inv_prm.map(lambda s: res[s][0]); V['rule'] = V.inv_prm.map(lambda s: res[s][1])
    V['disclosure'] = V.rcept_no.map(fil.rcept_dt).fillna('')
    num = lambda c: pd.to_numeric(c.str.replace(',', '', regex=False), errors='coerce')
    V['pct'] = num(V.trmend_blce_qota_rt); q = num(V.trmend_blce_qy); b = num(V.trmend_blce_acntbk_amount)
    V['invalid_pct'] = V.pct.notna() & ~V.pct.between(0, 100); V['held'] = (V.pct.gt(0) & V.pct.le(100)) | q.gt(0) | b.gt(0)
    ann = A[A.report_nm.str.match(r'^사업보고서\s*\(') & ~A.report_nm.str.contains('연장신고')].copy()
    ann['fy'] = ann.report_nm.str.extract(r'사업보고서\s*\((\d{4})\.')[0]
    observed = set(ann.loc[ann.fy.eq(YEAR) & ann.rcept_dt.le(C), 'corp_code']) & codes
    V['placeholder'] = V.inv_prm.str.strip().isin(PLACEHOLDER)
    keep = V.api_status.eq('000') & V.disclosure.ne('') & V.disclosure.le(C) & V.target.ne('') & ~V.invalid_pct & V.held \
        & V.src_corp_code.isin(observed) & V.target.isin(observed) & V.src_corp_code.ne(V.target)
    E = V[keep].drop_duplicates(['src_corp_code', 'target'])
    base = read(B / 'graphs/edges-2024-asof-20250430.csv')
    same = set(zip(base.src_corp_code, base.target)) == set(zip(E.src_corp_code, E.target))
    P = read(B / 'model-panel.csv'); T = P[P.split.eq('test')]
    iso = T[T.degree.eq('0')].corp_code.tolist(); nofin = T[T.degree.ne('0') & T.neighbor_financial_count.eq('0')].corp_code.tolist()
    deg = collections.Counter(list(E.src_corp_code) + list(E.target))
    # name history (rule ii): former names effective before C, unique, of universe firms
    former, nh_used = verified_former(codes); cur = set(names['norm'])
    unmatched = V[V.rule.isin(['unmatched', 'ambiguous']) & ~V.placeholder]
    rename_hits = collections.defaultdict(list)
    for r in unmatched.to_dict('records'):
        c = former.get(norm(r['inv_prm']), set())
        if len(c) == 1 and norm(r['inv_prm']) not in cur:
            t = next(iter(c))
            if t != r['src_corp_code'] and t in observed and r['src_corp_code'] in observed and r['held'] and r['disclosure'] and r['disclosure'] <= C:
                for k in (t, r['src_corp_code']): rename_hits[k].append(f"{r['src_corp_code']}->{t} via former name '{r['inv_prm']}' ({r['rcept_no']})")
    rows = []
    for k in iso:
        s = V[V.src_corp_code.eq(k)]
        real = s[~s.placeholder]
        ev = {'rows': len(s), 'real_rows': len(real), 'status': ','.join(sorted(set(s.api_status))) or 'no_rows',
              'disclosure': ','.join(sorted(set(s.disclosure)))}
        if k in rename_hits: cat = 'ii_name_match_failed'; ev['hits'] = rename_hits[k][:3]
        elif len(s) == 0 or not s.api_status.eq('000').any(): cat = 'i_not_obtained'
        elif s.disclosure.eq('').all() or (s.disclosure > C).all(): cat = 'i_not_obtained'; ev['note'] = 'filing date missing or after C'
        elif len(real) == 0: cat = 'i_or_iv_placeholder_only'; ev['placeholders'] = sorted(set(s.inv_prm))[:4]
        else:
            rl = collections.Counter(real.rule); ev['rules'] = dict(rl)
            m = real[real.target.ne('')]
            if len(m):
                why = []
                for r in m.to_dict('records'):
                    why.append('target_not_observed_by_C' if r['target'] not in observed else 'self' if r['target'] == k
                               else 'invalid_pct' if r['invalid_pct'] else 'not_held' if not r['held'] else 'late' if r['disclosure'] > C else 'other')
                ev['matched_dropped'] = dict(collections.Counter(why))
                cat = 'iii_matched_but_filtered' if len(m) == len(real) else 'iii_external_or_filtered'
            else: cat = 'iii_external_unmatched'
            ev['examples'] = list(real.inv_prm.head(3))
        rows.append(dict(corp_code=k, feature_fy=YEAR, category=cat, evidence=json.dumps(ev, ensure_ascii=False)))
    out = pd.DataFrame(rows); out.to_csv(R / 'data/processed/isolated_audit.csv', index=False)
    summ = {'baseline_graph_reproduced': same, 'edges_reproduced': len(E), 'edges_baseline': len(base),
            'isolated_test': len(iso), 'isolated_with_degree_in_reproduced_graph': sum(deg[k] > 0 for k in iso),
            'name_history_used': nh_used, 'categories': out.category.value_counts().to_dict(),
            'connected_no_fin_neighbor': len(nofin)}
    json.dump(summ, open(R / 'reports/isolated_audit.json', 'w'), indent=1, ensure_ascii=False); print(json.dumps(summ, ensure_ascii=False, indent=1))
    return V, E, observed, iso, nofin

if __name__ == '__main__': main()
