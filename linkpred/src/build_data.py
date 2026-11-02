"""Build link-prediction dataset per protocol.md (no API calls). Outputs under linkpred/data/."""
import collections, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
P = Path(__file__).resolve().parents[2]; D = P / 'linkpred/data'; D.mkdir(exist_ok=True)
sys.path.insert(0, str(P / 'src')); from isolated_audit import verified_former
read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
V = read(P / 'data/interim/invest_panel_raw.csv'); U = read(P / 'data/processed/universe_firms.csv'); O = read(P / 'data/interim/company_overview.csv'); A = read(P / 'data/interim/filings_A001.csv')
fil = A.drop_duplicates('rcept_no').set_index('rcept_no'); codes = set(U.corp_code)
suffix = re.compile(r'\(주\)|㈜|주식회사|\(유\)|유한회사|co\.?,?\s*ltd\.?|inc\.?|corp\.?|\s+', re.I)
norm = lambda s: suffix.sub('', s or '').replace('（', '(').replace('）', ')').strip().lower()
names = {k: collections.defaultdict(set) for k in ('exact', 'norm', 'stock')}
for r in O.to_dict('records'):
    if r['corp_code'] in codes:
        for k, v in [('exact', r['corp_name'].strip()), ('norm', norm(r['corp_name'])), ('stock', norm(r['stock_name']))]:
            if v: names[k][v].add(r['corp_code'])
def resolve(v):
    for k, n in [('exact', v.strip()), ('norm', norm(v)), ('stock', norm(v))]:
        c = names[k].get(n, set())
        if len(c) == 1: return next(iter(c))
        if len(c) > 1: return None   # ambiguous: never resolved via former names
    return ''
res = {s: resolve(s) for s in V.inv_prm.unique()}; cur = set(names['norm'])
num = lambda c: pd.to_numeric(c.str.replace(',', '', regex=False), errors='coerce')
V['pct'] = num(V.trmend_blce_qota_rt); q = num(V.trmend_blce_qy); b = num(V.trmend_blce_acntbk_amount)
V['ok'] = V.api_status.eq('000') & ~(V.pct.notna() & ~V.pct.between(0, 100)) & ((V.pct.gt(0) & V.pct.le(100)) | q.gt(0) | b.gt(0))
V['disc'] = V.rcept_no.map(fil.rcept_dt)
annual = A[A.report_nm.str.match(r'^사업보고서\s*\(') & ~A.report_nm.str.contains('연장신고')].copy()
annual['fy'] = annual.report_nm.str.extract(r'사업보고서\s*\((\d{4})\.')[0]
orig_first = annual[~annual.report_nm.str.contains('정정')].groupby(['corp_code', 'fy']).rcept_dt.min()
doc = read(P / 'data/processed/doc_recovered_edges_2024.csv')
C = lambda t: f'{t + 1}0430'
elig, G, H, Gpct = {}, {}, {}, {}
hist_dir = set(); log = []
for t in range(2019, 2026):
    c = C(t); elig[t] = set(annual.loc[annual.fy.eq(str(t)) & annual.rcept_dt.le(c), 'corp_code']) & codes
    former, _ = verified_former(codes, c)
    def tgt(v):
        r = res[v]
        if r or r is None: return r or ''
        cs = former.get(norm(v), set()); return next(iter(cs)) if len(cs) == 1 and norm(v) not in cur else ''
    s = V[V.bsns_year.eq(str(t)) & V.ok].copy(); s['t'] = s.inv_prm.map(tgt)
    s = s[s.t.ne('') & s.t.ne(s.src_corp_code) & s.src_corp_code.isin(codes) & s.t.isin(codes) & s.disc.ne('')]
    strict = s[s.disc.le(c) & s.src_corp_code.isin(elig[t]) & s.t.isin(elig[t])]
    ed = {}
    for a_, b_, p_ in zip(strict.src_corp_code, strict.t, strict.pct): ed[(a_, b_)] = np.nanmax([ed.get((a_, b_), np.nan), p_])
    nd = 0
    if t == 2024:
        for a_, b_, p_ in zip(doc.corp_code, doc.target, pd.to_numeric(doc.end_pct, errors='coerce')):
            if a_ in elig[t] and b_ in elig[t] and a_ != b_ and (a_, b_) not in ed: ed[(a_, b_)] = p_; nd += 1
    G[t] = ed
    # history (label side only): any rule, any disclosure date, incl. rule A + doc edges
    hist_dir |= set(zip(s.src_corp_code, s.t)) | set(ed)
    H[t] = {frozenset(p) for p in hist_dir}
    log.append(dict(fy=t, C=c, eligible=len(elig[t]), strict_directed=len(ed), doc_added=nd, history_undirected_cum=len(H[t])))
pd.DataFrame(log).to_csv(D / 'graph_years.csv', index=False)
pd.DataFrame([dict(fy=t, corp_code=c) for t in elig for c in sorted(elig[t])]).to_csv(D / 'eligible.csv', index=False)
# node attributes
F = read(P / 'data/interim/fin_multi_raw.csv'); F = F[F.fs_div.eq('OFS')]; F['disc'] = F.rcept_no.map(fil.rcept_dt)
fd = F.groupby(['corp_code', 'bsns_year']).disc.max()
fp = pd.read_csv(P / 'reruns/repaired-edges/fin_panel_rebuilt.csv', dtype={'corp_code': str, 'currencies': str})
FIELDS = ['log_assets', 'liabilities_assets', 'equity_assets', 'capital_assets', 'revenue_assets', 'operating_income_assets', 'net_income_assets', 'signed_log_revenue', 'signed_log_operating_income', 'current_operating_loss']
ov = O.drop_duplicates('corp_code').set_index('corp_code')
nodes = []
for t in range(2021, 2025):
    x = fp[fp.fy.eq(t) & fp.currencies.eq('KRW') & fp.assets.gt(0)].copy()
    x['fin_disc'] = [fd.get((cc, str(t)), '') for cc in x.corp_code]; x = x[x.fin_disc.ne('') & x.fin_disc.le(C(t))].set_index('corp_code')
    for cc in sorted(elig[t]):
        r = {'fy': t, 'corp_code': cc, 'industry': ov.induty_code.get(cc, '')[:2], 'market': ov.corp_cls.get(cc, ''), 'fin_disc': ''}
        if cc in x.index:
            z = x.loc[cc]; r['fin_disc'] = z.fin_disc; r['log_assets'] = np.log1p(z.assets)
            for n in ('liabilities', 'equity', 'capital', 'revenue', 'operating_income', 'net_income'): r[n + '_assets'] = z[n] / z.assets
            for n in ('revenue', 'operating_income'): r['signed_log_' + n] = np.sign(z[n]) * np.log1p(abs(z[n])) if pd.notna(z[n]) else np.nan
            r['current_operating_loss'] = float(z.operating_income < 0) if pd.notna(z.operating_income) else np.nan
        nodes.append(r)
NODES = pd.DataFrame(nodes).replace([np.inf, -np.inf], np.nan); NODES.to_csv(D / 'nodes.csv', index=False)
# pairs
rng = np.random.default_rng(20261103); pairs = []; split = {2021: 'train', 2022: 'train', 2023: 'validation', 2024: 'test'}; short = []
und = lambda ed: {frozenset(p) for p in ed}
for t in range(2021, 2025):
    E = elig[t] & elig[t + 1]; Eg = sorted(E); Ht = H[t]; Gn = und(G[t + 1])
    deg = collections.Counter(); [deg.update(p) for p in und(G[t])]
    new = collections.defaultdict(list)
    for (a_, b_), p_ in G[t + 1].items():
        k = frozenset((a_, b_))
        if k not in Ht and a_ in E and b_ in E: new[k].append((a_, b_, p_))
    blocked = Ht | Gn
    for k, rows in sorted(new.items(), key=lambda kv: sorted(kv[0])):
        u = min(a_ for a_, _, _ in rows); v = next(iter(k - {u})); pct = np.nanmax([p_ for *_, p_ in rows]) if rows else np.nan
        dirs = '+'.join(sorted(f'{a_}>{b_}' for a_, b_, _ in rows))
        pid = f'{t}-{u}-{v}'
        pairs.append(dict(pair_id=pid, fy=t, split=split[t], anchor=u, other=v, label=1, pct=pct, directions=dirs))
        cand = [w for w in Eg if w != u and frozenset((u, w)) not in blocked]
        pick = rng.choice(len(cand), size=min(100, len(cand)), replace=False)
        if len(cand) < 100: short.append(pid)
        for i in sorted(pick): pairs.append(dict(pair_id=pid, fy=t, split=split[t], anchor=u, other=cand[i], label=0, pct=np.nan, directions=''))
X = pd.DataFrame(pairs)
ind = NODES.set_index(['fy', 'corp_code']).industry
pos_pct = X[X.label.eq(1)].set_index('pair_id').pct
anc_fin = [ind.get((t, a_), '') in ('64', '65', '66') for t, a_ in zip(X.fy, X.anchor)]
X['investor_financial'] = anc_fin
X['subset_nonfin_ge5'] = (~X.investor_financial) & X.pair_id.map(pos_pct).ge(5)
degs = {t: collections.Counter(n for p in und(G[t]) for n in p) for t in range(2021, 2025)}
da = np.array([degs[t][a_] for t, a_ in zip(X.fy, X.anchor)]); do = np.array([degs[t][o_] for t, o_ in zip(X.fy, X.other)])
X['group'] = np.where((da > 0) & (do > 0), 'both_connected', np.where((da > 0) | (do > 0), 'one_isolated', 'both_isolated'))
X.to_csv(D / 'pairs.csv', index=False)
for t in range(2019, 2026):
    pd.DataFrame([dict(src=a_, dst=b_, pct=p_) for (a_, b_), p_ in G[t].items()]).to_csv(D / f'graph_{t}.csv', index=False)
pd.DataFrame([dict(fy=t, a=sorted(p)[0], b=sorted(p)[1]) for t in range(2021, 2025) for p in H[t]]).to_csv(D / 'history_undirected.csv', index=False)
json.dump({'short_negative_pairs': short}, open(D / 'build_notes.json', 'w'))
print(pd.DataFrame(log).to_string())
