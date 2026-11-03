"""共同投资图（描述性检查，只用训练年和验证年 FY2021–2023，不读测试年）。
两家上市公司在 FY t 的出资表里投资了同一家非上市主体（非上市公司、基金、组合等），就在它们之间记一条"共同投资"关系。
数据：data/interim/invest_panel_raw.csv，规则与 v3 一致：api_status=000、期末仍持有、披露日 ≤ C_t、投资方在 FY t 合格；
被投资方不能解析为上市范围内的公司（那些已在持股图 G_t 中）。不调用 API，不训练模型。
输出：linkpred/out_coinvest/（hub 分布、孤立企业覆盖、正负例对比）。
用法：python src/coinvest.py [--data data_v3] [--hub-cap 20]"""
import argparse, collections, json, re
from pathlib import Path
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--data', default='data_v3'); ap.add_argument('--hub-cap', type=int, default=20); ARGS = ap.parse_args()
L = Path(__file__).resolve().parents[1]; P = L.parent; D = L / ARGS.data; O = L / 'out_coinvest'; O.mkdir(exist_ok=True)
YEARS = (2021, 2022, 2023)            # 训练 2021→22、2022→23；验证 2023→24。测试年 2024 不在这里出现
rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
suffix = re.compile(r'\(주\)|㈜|주식회사|\(유\)|유한회사|co\.?,?\s*ltd\.?|inc\.?|corp\.?|\s+', re.I)
TOK = re.compile(r'\s*(\((?:비상장|상장|시장성|비시장성|해외 ?비상장|해외 ?상장|\*+\d*|주\d+)\)|\[(?:비상장|상장|해외 ?비상장|해외 ?상장)\])\s*$|\s+\*+\d*\s*$')
PLACE = {'', '-', '합계', '계', '소계', '총계', '해당사항없음', '해당사항 없음', '없음', '기타'}

def key(v):
    """把被投资方名称规范化成匹配键：去掉尾部标记、公司形态词、空白，转小写。不做模糊匹配。
    输入：原始名称；输出：匹配键（占位符返回空串）。"""
    v = (v or '').strip(); p = None
    while p != v: p = v; v = TOK.sub('', v).strip()
    k = suffix.sub('', v).replace('（', '(').replace('）', ')').lower()
    return '' if v in PLACE or k in PLACE or len(k) < 2 else k

def listed_keys():
    """上市范围内公司的名称键集合（现名、股票简称、经核实的曾用名），用来排除"其实是上市公司"的被投资方。"""
    U = set(rd(P / 'data/processed/universe_firms.csv').corp_code); O_ = rd(P / 'data/interim/company_overview.csv')
    ks = {key(n) for c, a, b in zip(O_.corp_code, O_.corp_name, O_.stock_name) if c in U for n in (a, b)}
    import sys; sys.path.insert(0, str(P / 'src')); from isolated_audit import verified_former
    ks |= {key(n) for n in verified_former(U, '99999999')[0]}
    return ks - {''}

def build_year(V, t, elig, LK):
    """构造 FY t 的"上市投资方 → 非上市主体"二部图。
    输入：出资行、年份、FY t 合格名单、上市名称键；输出：dict 主体键 → 投资方集合。"""
    c = f'{t + 1}0430'
    s = V[V.bsns_year.eq(str(t)) & V.ok & V.disc.ne('') & V.disc.le(c) & V.src_corp_code.isin(elig)]
    s = s[s.k.ne('') & ~s.k.isin(LK)]
    B = collections.defaultdict(set)
    for a, k in zip(s.src_corp_code, s.k): B[k].add(a)
    return B

def project(B, cap):
    """把二部图投影成上市公司之间的共同投资关系。
    输入：二部图、hub 上限 cap（被超过 cap 家上市公司共同持有的主体视为基础设施类 hub，如공제조합，剔除）；
    输出：(邻接 dict 公司 → {邻居: 共同主体数}, Adamic–Adar 权重 dict (u,v) → Σ 1/log(主体持有家数))。"""
    adj = collections.defaultdict(collections.Counter); aa = collections.Counter()
    for k, inv in B.items():
        n = len(inv)
        if n < 2 or (cap and n > cap): continue
        inv = sorted(inv); w = 1 / np.log(n) if n > 1 else 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = inv[i], inv[j]; adj[a][b] += 1; adj[b][a] += 1; aa[frozenset((a, b))] += w
    return adj, aa

def within_auc(df, col):
    """组内 AUC：每个正例在自己的 101 个候选中按 col 排序，并列取期望秩。输入：点对表、分数列；输出：均值。"""
    pos = df[df.label == 1].set_index('pair_id')[col]; neg = df[df.label == 0]
    x = neg.pair_id.map(pos); n = neg.groupby('pair_id').size()
    return float((((neg[col] < x).groupby(neg.pair_id).sum() + (neg[col] == x).groupby(neg.pair_id).sum() / 2) / n).mean())

V = rd(P / 'data/interim/invest_panel_raw.csv'); A = rd(P / 'data/interim/filings_A001.csv')
fil = A.drop_duplicates('rcept_no').set_index('rcept_no')
num = lambda c: pd.to_numeric(c.str.replace(',', '', regex=False), errors='coerce')
pct, q, b = num(V.trmend_blce_qota_rt), num(V.trmend_blce_qy), num(V.trmend_blce_acntbk_amount)
V['ok'] = V.api_status.eq('000') & ~(pct.notna() & ~pct.between(0, 100)) & ((pct.gt(0) & pct.le(100)) | q.gt(0) | b.gt(0))
V['disc'] = V.rcept_no.map(fil.rcept_dt).fillna(''); V['k'] = V.inv_prm.map(key)
E = rd(D / 'eligible.csv'); X = pd.read_csv(D / 'pairs.csv', dtype={'anchor': str, 'other': str})
X = X[X.fy.isin(YEARS)].copy()                      # 只取训练年和验证年的点对
LK = listed_keys(); res = {'hub_cap': ARGS.hub_cap, 'years': list(YEARS)}
hub_rows = []
for t in YEARS:
    elig = set(E[E.fy == str(t)].corp_code); B = build_year(V, t, elig, LK)
    sizes = pd.Series({k: len(v) for k, v in B.items()})
    res[f'bipartite_{t}'] = dict(entities=int(len(sizes)), shared_by_2plus=int((sizes >= 2).sum()), over_cap=int((sizes > ARGS.hub_cap).sum()),
                                 size_quantiles={str(qq): float(sizes.quantile(qq)) for qq in (.5, .9, .99, 1)})
    for k, n in sizes.sort_values(ascending=False).head(15).items(): hub_rows.append(dict(fy=t, entity_key=k, n_listed_holders=int(n)))
    G = rd(D / f'graph_{t}.csv'); conn = set(G.src) | set(G.dst); iso = elig - conn
    for cap in (ARGS.hub_cap, 0):
        adj, aa = project(B, cap); tag = f'cap{cap}' if cap else 'nocap'
        res[f'isolated_{t}_{tag}'] = dict(isolated=len(iso), with_coinvest_neighbour=sum(1 for c in iso if adj.get(c)),
                                          connected_with_coinvest_neighbour=sum(1 for c in conn & elig if adj.get(c)))
        m = X.fy == t
        X.loc[m, f'ci_{tag}'] = [adj.get(a, {}).get(o, 0) for a, o in zip(X.anchor[m], X.other[m])]
        X.loc[m, f'ciaa_{tag}'] = [aa.get(frozenset((a, o)), 0.) for a, o in zip(X.anchor[m], X.other[m])]
        # 增强图 = G_t ∪ 共同投资关系；在增强图上算共同邻居数和 Adamic–Adar，并与只用 G_t 的结果对比
        Ng = collections.defaultdict(set)
        for a, b_ in zip(G.src, G.dst): Ng[a].add(b_); Ng[b_].add(a)
        Na = collections.defaultdict(set, {k: set(v) for k, v in Ng.items()})
        for a, nb in adj.items(): Na[a] |= set(nb)
        def cn_aa(N_, a, o):
            """在给定邻接上算 (共同邻居数, Adamic–Adar)。"""
            c_ = N_.get(a, set()) & N_.get(o, set()); return len(c_), sum(1 / np.log(len(N_[z])) for z in c_ if len(N_[z]) > 1)
        h_g = [cn_aa(Ng, a, o) for a, o in zip(X.anchor[m], X.other[m])]; h_a = [cn_aa(Na, a, o) for a, o in zip(X.anchor[m], X.other[m])]
        X.loc[m, f'cn_g_{tag}'] = [x[0] for x in h_g]; X.loc[m, f'aa_g_{tag}'] = [x[1] for x in h_g]
        X.loc[m, f'cn_aug_{tag}'] = [x[0] for x in h_a]; X.loc[m, f'aa_aug_{tag}'] = [x[1] for x in h_a]
        # 直接共同投资 + 增强图 AA 的组合分数（只是描述用的单一分数，不是模型）
        X.loc[m, f'combo_{tag}'] = X.loc[m, f'ciaa_{tag}'] * 10 + X.loc[m, f'aa_aug_{tag}']
pd.DataFrame(hub_rows).to_csv(O / 'top_hubs.csv', index=False)
# 正负例对比：按年份、按"正例所在组"（候选集保持完整的 101 个）
pgroup = X[X.label == 1].set_index('pair_id').group
X['pos_group'] = X.pair_id.map(pgroup)
rows = []
for (t, sp), g in X.groupby(['fy', 'split']):
    for grp, h in [('all', g)] + list(g.groupby('pos_group')):
        for tag in (f'cap{ARGS.hub_cap}', 'nocap'):
            p, n = h[h.label == 1], h[h.label == 0]
            r = dict(fy=t, split=sp, pos_group=grp, hub=tag, n_pos=len(p))
            for f, lab in (('ci', 'direct'), ('cn_g', 'cn_Gt'), ('cn_aug', 'cn_aug')):
                rp, rn = float((p[f'{f}_{tag}'] > 0).mean()), float((n[f'{f}_{tag}'] > 0).mean())
                r[f'pos_{lab}>0'] = round(rp, 3); r[f'neg_{lab}>0'] = round(rn, 4)
            for f in ('aa_g', 'aa_aug', 'ciaa', 'combo'): r[f'wAUC_{f}'] = round(within_auc(h, f'{f}_{tag}'), 3)
            rows.append(r)
R = pd.DataFrame(rows); R.to_csv(O / 'pos_vs_neg.csv', index=False)
json.dump(res, open(O / 'summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False)); print(R.to_string(index=False))
