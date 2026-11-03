"""度偏差论文研究（protocol.md A7）的公共部分：读取 data_v3、构造节点特征、共同投资图、组内指标与 bootstrap。
只读数据，不训练模型；被 paper_run.py 调用。"""
import collections, re, sys
from pathlib import Path
import numpy as np, pandas as pd

L = Path(__file__).resolve().parents[1]; P = L.parent; D = L / 'data_v3'
YEARS = (2021, 2022, 2023, 2024)            # 特征年 t；标签是 t+1 年的新边
FI = ['log_assets', 'liabilities_assets', 'equity_assets', 'capital_assets', 'revenue_assets', 'operating_income_assets',
      'net_income_assets', 'signed_log_revenue', 'signed_log_operating_income', 'current_operating_loss']
rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)

def load_pairs():
    """读取正例和负例点对。输出：DataFrame（fy、split、anchor、other、label、group 等），并加上正例所在组 pos_group。"""
    X = pd.read_csv(D / 'pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str})
    X['pos_group'] = X.pair_id.map(X[X.label == 1].set_index('pair_id').group)
    return X

def load_graph(t):
    """读取 FY t 的持股图 G_t（截止日 C_t 前披露）。输出：无向边列表 [(a, b)]。"""
    g = rd(D / f'graph_{t}.csv'); return [(a, b) for a, b in zip(g.src, g.dst) if a != b]

# ---------- 共同投资图（规则同 src/coinvest.py；结果缓存为 data_v3/coinvest_t.csv） ----------
_suffix = re.compile(r'\(주\)|㈜|주식회사|\(유\)|유한회사|co\.?,?\s*ltd\.?|inc\.?|corp\.?|\s+', re.I)
_tok = re.compile(r'\s*(\((?:비상장|상장|시장성|비시장성|해외 ?비상장|해외 ?상장|\*+\d*|주\d+)\)|\[(?:비상장|상장|해외 ?비상장|해외 ?상장)\])\s*$|\s+\*+\d*\s*$')
_place = {'', '-', '합계', '계', '소계', '총계', '해당사항없음', '해당사항 없음', '없음', '기타'}

def _key(v):
    """被投资方名称 → 匹配键（去尾部标记、公司形态词、空白，小写；占位符返回空串）。"""
    v = (v or '').strip(); p = None
    while p != v: p = v; v = _tok.sub('', v).strip()
    k = _suffix.sub('', v).replace('（', '(').replace('）', ')').lower()
    return '' if v in _place or k in _place or len(k) < 2 else k

def coinvest_edges(t, cap=20):
    """FY t 的共同投资关系：两家合格上市公司共同持有同一非上市主体（被超过 cap 家共同持有的主体视为 hub 剔除）。
    输入：年份、hub 上限；输出：无向边列表 [(a, b)]。第一次计算后缓存到 data_v3/coinvest_t.csv。"""
    f = D / f'coinvest_{t}.csv'
    if f.exists():
        g = rd(f); return list(zip(g.a, g.b))
    V = rd(P / 'data/interim/invest_panel_raw.csv'); A = rd(P / 'data/interim/filings_A001.csv')
    fil = A.drop_duplicates('rcept_no').set_index('rcept_no')
    num = lambda c: pd.to_numeric(c.str.replace(',', '', regex=False), errors='coerce')
    pct, q, b = num(V.trmend_blce_qota_rt), num(V.trmend_blce_qy), num(V.trmend_blce_acntbk_amount)
    V['ok'] = V.api_status.eq('000') & ~(pct.notna() & ~pct.between(0, 100)) & ((pct.gt(0) & pct.le(100)) | q.gt(0) | b.gt(0))
    V['disc'] = V.rcept_no.map(fil.rcept_dt).fillna('')
    U = set(rd(P / 'data/processed/universe_firms.csv').corp_code); O = rd(P / 'data/interim/company_overview.csv')
    sys.path.insert(0, str(P / 'src')); from isolated_audit import verified_former
    LK = {_key(n) for c, a, b_ in zip(O.corp_code, O.corp_name, O.stock_name) if c in U for n in (a, b_)} | {_key(n) for n in verified_former(U, '99999999')[0]}
    LK -= {''}
    E = rd(D / 'eligible.csv'); elig = set(E[E.fy == str(t)].corp_code)
    s = V[V.bsns_year.eq(str(t)) & V.ok & V.disc.ne('') & V.disc.le(f'{t + 1}0430') & V.src_corp_code.isin(elig)].copy()
    s['k'] = s.inv_prm.map(_key); s = s[s.k.ne('') & ~s.k.isin(LK)]
    B = collections.defaultdict(set)
    for a, k in zip(s.src_corp_code, s.k): B[k].add(a)
    ed = set()
    for inv in B.values():
        if 2 <= len(inv) <= cap:
            inv = sorted(inv); ed |= {(inv[i], inv[j]) for i in range(len(inv)) for j in range(i + 1, len(inv))}
    pd.DataFrame(sorted(ed), columns=['a', 'b']).to_csv(f, index=False)
    return sorted(ed)

# ---------- 节点特征（与 M1、m3_sage.py 相同口径；统计量只用训练年 2021、2022） ----------
def node_tables(with_ci=False):
    """为每个特征年构造节点集合、标准化特征矩阵和无向边。
    输入：是否把共同投资图并入结构图；输出：dict 年份 → {ids, idx, x(np), edges(list of (i,j)), deg(np)}。"""
    N = pd.read_csv(D / 'nodes.csv', dtype={'corp_code': str, 'industry': str, 'market': str, 'fin_disc': str}, keep_default_na=False, na_values=[''])
    tr = N[N.fy.isin([2021, 2022])]
    lo, hi = tr[FI].quantile(.005), tr[FI].quantile(.995); med = tr[FI].clip(lo, hi, axis=1).median()
    topind = tr.industry.value_counts().index[:20].tolist()
    out = {}
    for t in YEARS:
        E = load_graph(t) + (coinvest_edges(t) if with_ci else [])
        Nt = N[N.fy == t].set_index('corp_code'); ids = sorted(set(Nt.index) | {n for e in E for n in e})
        M = Nt.reindex(ids); cols = []
        for c in FI:
            cols.append(M[c].isna().astype(float).values); cols.append(M[c].clip(lo[c], hi[c]).fillna(med[c]).values)
        ind = M.industry.fillna('').values; mk = M.market.fillna('').values
        for i in topind: cols.append((ind == i).astype(float))
        cols.append((~np.isin(ind, topind)).astype(float))
        for m in ('Y', 'K'): cols.append((mk == m).astype(float))
        idx = {c: i for i, c in enumerate(ids)}
        e = sorted({(min(idx[a], idx[b]), max(idx[a], idx[b])) for a, b in E if a != b})
        deg = np.zeros(len(ids));
        for i, j in e: deg[i] += 1; deg[j] += 1
        out[t] = dict(ids=ids, idx=idx, F=np.vstack(cols).T, edges=e, deg=deg)
    tr_F = np.vstack([out[t]['F'] for t in (2021, 2022)]); mu, sd = tr_F.mean(0), tr_F.std(0); sd[sd == 0] = 1
    for t in YEARS: out[t]['x'] = ((out[t]['F'] - mu) / sd).astype(np.float32)
    return out

# ---------- 评估 ----------
def within_metrics(df, col):
    """每个正例在自己的 101 个候选中的组内指标（并列取期望秩）。
    输入：含 pair_id、label、anchor、pos_group 和分数列 col 的 DataFrame；输出：每个正例一行（AUC、RR、H10）。"""
    pos = df[df.label == 1].set_index('pair_id'); neg = df[df.label == 0]
    x = neg.pair_id.map(pos[col]); g = neg.groupby('pair_id')
    gt = (neg[col] > x).groupby(neg.pair_id).sum(); lt = (neg[col] < x).groupby(neg.pair_id).sum(); eq = (neg[col] == x).groupby(neg.pair_id).sum(); n = g.size()
    rank = gt + 1 + eq / 2
    W = pd.DataFrame({'AUC': (lt + eq / 2) / n, 'RR': 1 / rank, 'H10': (rank <= 10).astype(float)})
    return W.join(pos[['anchor', 'pos_group']])

def boot_ci(W, fn, seed=20261103, B=1000):
    """按投资方聚类的配对 bootstrap。输入：每个正例一行的表 W（含 anchor）、统计量函数 fn(W子表)→数；输出：(点估计, 下限, 上限)。"""
    ua = np.unique(W.anchor.values); ix = {a: np.where(W.anchor.values == a)[0] for a in ua}
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        d = rng.choice(ua, len(ua)); v.append(fn(W.iloc[np.concatenate([ix[a] for a in d])]))
    v = np.array(v, dtype=float)
    return float(fn(W)), float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))
