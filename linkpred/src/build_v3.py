"""数据版本 v3：在 build_v2.py（A2 名称规则 + A4 只用 API 规则）的基础上做三处修改，输出到 --out（默认 data_v3/）。
不调用 API；不修改任何已有数据目录。

修改（每项可单独关闭，三项都关时输出应与 data_a2_apionly/ 逐字节一致，用于回归检查）：
  --names-any yes  名称按"当时有效的名称"解析：曾用名表不再要求改名公告在 C_t 之前。
                   改名公告只用来确认"这个旧名称就是这家公司"，不带入任何业务信息；
                   投资行本身仍须在 C_t 之前披露（as-of 规则不变）。
  --a2b yes        在 A2 尾部标记清单上，再去掉不带括号的脚注标记，例如 "TKG휴켐스 *4"。
  --newness yes    "新边"必须真的是新关系：t+1 年的出资行显示期初没有持有（期初数量、账面价值、比例都为 0 或空），
                   或首次取得日在 t+1 财年内。期初已持有且首次取得在 t+1 财年之前的点对，记为"早已存在"：
                   不作正例，也不作负例，单独输出到 preexisting_pairs.csv。
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--out', default='data_v3'); ap.add_argument('--names-any', default='yes'); ap.add_argument('--a2b', default='yes'); ap.add_argument('--newness', default='yes'); ap.add_argument('--first-year', type=int, default=2021)
ARGS = ap.parse_args()
FY = list(range(ARGS.first_year, 2025))   # 特征年；默认 2021–2024。--first-year 2019 时另加 2019→20、2020→21（split=dev_early，用独立的随机流，2021 起的负例与默认版本完全相同）
import collections, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
P = Path(__file__).resolve().parents[2]; D = P / 'linkpred' / ARGS.out; D.mkdir(exist_ok=True)
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
A2_TOK = re.compile(r'\s*(\((?:비상장|상장|시장성|비시장성|해외 ?비상장|해외 ?상장|\*+\d*|주\d+)\)|\[(?:비상장|상장|해외 ?비상장|해외 ?상장)\])\s*$')
A2B_TOK = re.compile(r'\s+\*+\d*\s*$')      # v3：不带括号、前面有空白的脚注标记，如 " *4"

def a2(v):
    """去掉名称尾部的上市状态/脚注标记（反复去，直到不再变化）。
    输入：原始被投资方名称字符串；输出：去掉标记后的名称。只删除清单中完全匹配的标记，不做模糊处理。"""
    v = v.strip(); prev = None
    while prev != v:
        prev = v; v = A2_TOK.sub('', v).strip()
        if ARGS.a2b == 'yes': v = A2B_TOK.sub('', v).strip()
    return v

def resolve(v):
    """按现有规则把名称解析为 corp_code：精确名 → 规范化名 → 股票简称，必须唯一。
    输入：被投资方名称；输出：corp_code（唯一匹配）、None（多重匹配，拒绝）或 ''（未匹配）。"""
    v = a2(v)
    for k, n in [('exact', v.strip()), ('norm', norm(v)), ('stock', norm(v))]:
        c = names[k].get(n, set())
        if len(c) == 1: return next(iter(c))
        if len(c) > 1: return None
    return ''
res = {s: resolve(s) for s in V.inv_prm.unique()}; cur = set(names['norm'])
num = lambda c: pd.to_numeric(c.str.replace(',', '', regex=False), errors='coerce')
V['pct'] = num(V.trmend_blce_qota_rt); q = num(V.trmend_blce_qy); b = num(V.trmend_blce_acntbk_amount)
V['ok'] = V.api_status.eq('000') & ~(V.pct.notna() & ~V.pct.between(0, 100)) & ((V.pct.gt(0) & V.pct.le(100)) | q.gt(0) | b.gt(0))
V['disc'] = V.rcept_no.map(fil.rcept_dt)
# 期初余额（任一为正即视为期初已持有）
V['open_pos'] = (num(V.bsis_blce_qy).fillna(0) > 0) | (num(V.bsis_blce_acntbk_amount).fillna(0) > 0) | (num(V.bsis_blce_qota_rt).fillna(0) > 0)
annual = A[A.report_nm.str.match(r'^사업보고서\s*\(') & ~A.report_nm.str.contains('연장신고')].copy()
annual['fy'] = annual.report_nm.str.extract(r'사업보고서\s*\((\d{4})\.')[0]
C = lambda t: f'{t + 1}0430'
former_any, _ = verified_former(codes, '99999999') if ARGS.names_any == 'yes' else ({}, False)

def parse_date(s):
    """把首次取得日解析成 (年, 月, 日)；只有年份时月、日为 None；无法解析返回 None。
    输入：如 '1999.05.21'、'2021-1-26'、'2015년'、'-'；输出：元组或 None。"""
    m = re.search(r'(\d{4})\D+(\d{1,2})\D+(\d{1,2})', s or '')
    if m: return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r'(19|20)\d{2}', s or '')
    return (int(m.group(0)), None, None) if m else None

def row_newness(acq, stlm, open_pos, fy1):
    """判断 t+1 年的一条出资行是否说明这是新关系。
    输入：首次取得日字符串、结算日（财年末，YYYY-MM-DD）、期初是否持有、t+1 年份。
    输出：'new_opening_zero'（期初未持有）、'new_acquired_in_year'（期初有余额但首次取得在本财年内，字段自相矛盾，按新关系计并标记）、
          'preexisting'（期初持有且首次取得早于本财年或未知）。
    财年区间 = (结算日前一年的次日, 结算日]；结算日缺失时按 12 月 31 日。"""
    if not open_pos: return 'new_opening_zero'
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', stlm or ''); end = tuple(map(int, m.groups())) if m else (fy1, 12, 31)
    start = (end[0] - 1, end[1], end[2])          # 区间左端（不含）
    d = parse_date(acq)
    if d is None: return 'preexisting'
    if d[1] is None: return 'new_acquired_in_year' if d[0] > start[0] and d[0] <= end[0] and end[1] == 12 else 'preexisting'
    return 'new_acquired_in_year' if start < d <= end else 'preexisting'

elig, G, H, ROWS = {}, {}, {}, {}
hist_dir = set(); log = []; FLOW = {}
for t in range(2019, 2026):
    c = C(t); elig[t] = set(annual.loc[annual.fy.eq(str(t)) & annual.rcept_dt.le(c), 'corp_code']) & codes
    former = former_any if ARGS.names_any == 'yes' else verified_former(codes, c)[0]
    def tgt(v):
        """把一个名称解析为目标 corp_code：先用现有规则，失败时再用经核实的曾用名（唯一且不与现名冲突）。"""
        r = res[v]
        if r or r is None: return r or ''
        key = norm(a2(v)) if ARGS.a2b == 'yes' else norm(v)      # v3：曾用名查找也先去掉尾部标记；v2 用原名
        cs = former.get(key, set())
        return next(iter(cs)) if len(cs) == 1 and key not in cur else ''
    s = V[V.bsns_year.eq(str(t)) & V.ok].copy(); s['t'] = s.inv_prm.map(tgt)
    s = s[s.t.ne('') & s.t.ne(s.src_corp_code) & s.src_corp_code.isin(codes) & s.t.isin(codes) & s.disc.ne('')]
    strict = s[s.disc.le(c) & s.src_corp_code.isin(elig[t]) & s.t.isin(elig[t])]
    ed = {}
    for a_, b_, p_ in zip(strict.src_corp_code, strict.t, strict.pct): ed[(a_, b_)] = np.nanmax([ed.get((a_, b_), np.nan), p_])
    G[t] = ed; ROWS[t] = strict
    FLOW.setdefault(t, {}).update(resolved_held_rows=len(s), rows_disc_le_C=int(s.disc.le(c).sum()), rows_both_elig=len(strict), directed_pairs=len(ed), undirected_pairs=len({frozenset(k) for k in ed}))
    hist_dir |= set(zip(s.src_corp_code, s.t)) | set(ed)        # 历史集合（只用于标签）：任何规则、任何披露日
    H[t] = {frozenset(p) for p in hist_dir}
    log.append(dict(fy=t, C=c, eligible=len(elig[t]), strict_directed=len(ed), doc_added=0, history_undirected_cum=len(H[t])))
pd.DataFrame(log).to_csv(D / 'graph_years.csv', index=False)
pd.DataFrame([dict(fy=t, corp_code=c) for t in elig for c in sorted(elig[t])]).to_csv(D / 'eligible.csv', index=False)
# ---------- 节点属性（与 v2 相同） ----------
F = read(P / 'data/interim/fin_multi_raw.csv'); F = F[F.fs_div.eq('OFS')]; F['disc'] = F.rcept_no.map(fil.rcept_dt)
fd = F.groupby(['corp_code', 'bsns_year']).disc.max()
fp = pd.read_csv(P / 'reruns/repaired-edges/fin_panel_rebuilt.csv', dtype={'corp_code': str, 'currencies': str})
ov = O.drop_duplicates('corp_code').set_index('corp_code')
nodes = []
for t in FY:
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
# ---------- 点对：正例 + 每个正例 100 个负例 ----------
rng = np.random.default_rng(20261103); rng_early = np.random.default_rng(20261103 + 1000); pairs = []
split = {2019: 'dev_early', 2020: 'dev_early', 2021: 'train', 2022: 'train', 2023: 'validation', 2024: 'test'}; short = []
und = lambda ed: {frozenset(p) for p in ed}
firsts = []
for t in range(2020, 2026):
    Gu = und(G[t]); firsts.append(dict(fy=t, total_undirected=len(Gu), first_appearing=len(Gu - H[t - 1]), first_appearing_doc_only=0))
pd.DataFrame(firsts).to_csv(D / 'first_appearance_by_fy.csv', index=False)

def pair_newness(t, rows):
    """判断一个首次出现的点对在 t+1 年是否真是新关系。
    输入：年份 t、该点对在 t+1 年的有向边列表 [(投资方, 被投资方, 比例)]；
    输出：(结论, 依据)。任何一条出资行显示'早已存在'即判为 preexisting（关系在 t 年末已存在，只是 t 年没有被观测到）。"""
    R1 = ROWS[t + 1]; ev = []
    for a_, b_, _ in rows:
        rr = R1[(R1.src_corp_code == a_) & (R1.t == b_)]
        ev += [row_newness(acq, st, op, t + 1) for acq, st, op in zip(rr.frst_acqs_de, rr.stlm_dt, rr.open_pos)]
    if 'preexisting' in ev: return 'preexisting', ';'.join(sorted(set(ev)))
    return ('new_acquired_in_year' if 'new_acquired_in_year' in ev else 'new_opening_zero'), ';'.join(sorted(set(ev)))

flow = []; PRE = []
for t in FY:
    E = elig[t] & elig[t + 1]; Eg = sorted(E); Ht = H[t]; Gn = und(G[t + 1])
    new = collections.defaultdict(list)
    for (a_, b_), p_ in G[t + 1].items():
        k = frozenset((a_, b_))
        if k not in Ht and a_ in E and b_ in E: new[k].append((a_, b_, p_))
    fa = Gn - Ht; bothE = {k for k in fa if k <= E}
    verdict = {k: pair_newness(t, rows) if ARGS.newness == 'yes' else ('new_unchecked', '') for k, rows in new.items()}
    pre = {k for k, (v, _) in verdict.items() if v == 'preexisting'}
    for k in sorted(pre, key=sorted):
        PRE.append(dict(fy=t, a=sorted(k)[0], b=sorted(k)[1], evidence=verdict[k][1], directions='+'.join(sorted(f'{a_}>{b_}' for a_, b_, _ in new[k]))))
    hasfin = set(NODES[NODES.fy.eq(t) & NODES.fin_disc.ne('')].corp_code)
    flow.append(dict(split=f'{t}->{t+1}', **{f'G{t+1}_' + k: v for k, v in FLOW[t + 1].items()}, removed_not_first_appearance=len(Gn) - len(fa), first_appearing=len(fa),
        removed_endpoint_not_eligible_FY_t=len(fa) - len(bothE), removed_preexisting=len(pre), positives=len(bothE) - len(pre),
        positives_doc_only=0, pos_both_fin_available=sum(k <= hasfin for k in bothE - pre), pos_investor_nonfin_ge5=None))
    blocked = Ht | Gn            # 早已存在的点对属于 G_{t+1}，因此自动不会被抽成负例
    for k, rows in sorted(new.items(), key=lambda kv: sorted(kv[0])):
        if k in pre: continue
        u = min(a_ for a_, _, _ in rows); v = next(iter(k - {u})); pct = np.nanmax([p_ for *_, p_ in rows]) if rows else np.nan
        dirs = '+'.join(sorted(f'{a_}>{b_}' for a_, b_, _ in rows))
        pid = f'{t}-{u}-{v}'
        rec = dict(pair_id=pid, fy=t, split=split[t], anchor=u, other=v, label=1, pct=pct, directions=dirs)
        if ARGS.newness == 'yes': rec['new_evidence'] = verdict[k][0]
        pairs.append(rec)
        cand = [w for w in Eg if w != u and frozenset((u, w)) not in blocked]
        pick = (rng if t >= 2021 else rng_early).choice(len(cand), size=min(100, len(cand)), replace=False)
        if len(cand) < 100: short.append(pid)
        for i in sorted(pick):
            rec = dict(pair_id=pid, fy=t, split=split[t], anchor=u, other=cand[i], label=0, pct=np.nan, directions='')
            if ARGS.newness == 'yes': rec['new_evidence'] = ''
            pairs.append(rec)
X = pd.DataFrame(pairs)
ind = NODES.set_index(['fy', 'corp_code']).industry
pos_pct = X[X.label.eq(1)].set_index('pair_id').pct
X['investor_financial'] = [ind.get((t, a_), '') in ('64', '65', '66') for t, a_ in zip(X.fy, X.anchor)]
X['subset_nonfin_ge5'] = (~X.investor_financial) & X.pair_id.map(pos_pct).ge(5)
degs = {t: collections.Counter(n for p in und(G[t]) for n in p) for t in FY}
da = np.array([degs[t][a_] for t, a_ in zip(X.fy, X.anchor)]); do = np.array([degs[t][o_] for t, o_ in zip(X.fy, X.other)])
X['group'] = np.where((da > 0) & (do > 0), 'both_connected', np.where((da > 0) | (do > 0), 'one_isolated', 'both_isolated'))
X['also_in_2024_doc_set'] = 0
X.to_csv(D / 'pairs.csv', index=False)
FL = pd.DataFrame(flow); pp = X[X.label.eq(1)]
FL['pos_investor_financial'] = pp.groupby('fy').investor_financial.sum().values; FL['pos_investor_nonfin_ge5'] = pp.groupby('fy').subset_nonfin_ge5.sum().values
FL['pos_with_pct_missing'] = pp.groupby('fy').pct.apply(lambda s: s.isna().sum()).values
if ARGS.newness == 'no': FL = FL.drop(columns='removed_preexisting')
FL.T.to_csv(D / 'sample_flow.csv')
json.dump(dict(doc_only_pairs_2024=0), open(D / 'doc_only.json', 'w'))
for t in range(2019, 2026):
    pd.DataFrame([dict(src=a_, dst=b_, pct=p_) for (a_, b_), p_ in G[t].items()]).to_csv(D / f'graph_{t}.csv', index=False)
pd.DataFrame([dict(fy=t, a=sorted(p)[0], b=sorted(p)[1]) for t in FY for p in H[t]]).to_csv(D / 'history_undirected.csv', index=False)
if ARGS.newness == 'yes': pd.DataFrame(PRE, columns=['fy', 'a', 'b', 'evidence', 'directions']).to_csv(D / 'preexisting_pairs.csv', index=False)
json.dump({'short_negative_pairs': short}, open(D / 'build_notes.json', 'w'))
print(pd.DataFrame(log).to_string()); print(FL[['split', 'first_appearing', 'positives'] + (['removed_preexisting'] if ARGS.newness == 'yes' else [])].to_string())
