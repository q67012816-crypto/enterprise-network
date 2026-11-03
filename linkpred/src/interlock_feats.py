"""第 3 项：董事兼任（연결 이사）点对特征。只算 fy 2019–2023（训练年和验证年；不计算测试年 fy 2024）。
数据：data/interim/exec_panel_raw.csv（src/exec_panel.py 下载的 exctvSttus，即年报"임원현황"，含등기임원和미등기임원）。
同一个人 = 姓名（去空白）+ 出生年月（YYYYMM）；缺出生年月的行丢弃。FY t 只用 rcept_no 日期 ≤ C_t（t+1 年 4 月 30 日）的年报。
特征（a = 投资方，o = 候选企业）：
  il_direct    FY t 两家公司共有的董事人数（log1p）
  il_inside    其中至少在一家公司是执行人员的人数（log1p）。执行人员 = 不是사외이사、也不是감사；
               各年口径统一：2019–2020 年 rgist_exctv_at 只写등기/미등기，所以"사외""감사"同时看 rgist_exctv_at 和 ofcps
  il_2hop      FY t 与 a、o 都共有董事的第三家公司数（log1p）
  il_hist      s ≤ t 的任意年份在 a 任过职、且在任意年份 s' ≤ t 在 o 任过职的人数，去掉 FY t 的直接兼任（log1p；人员流动/过去兼任）
  il_has_a / il_has_o   该公司 FY t 是否有可用的임원현황（覆盖指示）
输出：out_dev/interlock_feats.csv（索引与 pairs.csv 行号一致）。用法：python src/interlock_feats.py [--lenient]
--lenient（敏感性分析）：保留 C_t 之后才提交的更正年报（API 只返回最新版本），输出 interlock_feats_lenient.csv。"""
import collections, re, sys
from pathlib import Path
import numpy as np, pandas as pd

L = Path(__file__).resolve().parents[1]; P0 = L.parent; D = L / 'data_v3x'; O = L / 'out_dev'
YEARS = range(2019, 2024)
IL = ['il_direct', 'il_inside', 'il_2hop', 'il_hist', 'il_has_a', 'il_has_o']

def persons():
    """读取임원현황，返回 dict 年份 → dict 公司 → {人: 是否사내이사}，并打印覆盖情况。"""
    E = pd.read_csv(P0 / 'data/interim/exec_panel_raw.csv', dtype=str, keep_default_na=False)
    ok = E.api_status.eq('000') & E.nm.ne('')
    b = E.birth_ym.str.replace(r'\D', '', regex=True)
    role = E.rgist_exctv_at + '|' + E.ofcps
    E = E.assign(pid=E.nm.str.replace(r'\s+', '', regex=True) + '|' + b, inside=~role.str.contains('사외|감사'),
                 rdate=E.rcept_no.str[:8], fy=pd.to_numeric(E.bsns_year, errors='coerce'))
    E = E[ok & b.str.len().eq(6)]
    if '--lenient' not in sys.argv: E = E[E.rdate.le((E.fy + 1).astype(int).astype(str) + '0430')]
    out = {}
    for t, g in E.groupby('fy'):
        d = collections.defaultdict(dict)
        for c, p, ins in zip(g.src_corp_code, g.pid, g.inside): d[c][p] = d[c].get(p, False) or ins
        out[int(t)] = d
        print(t, 'firms', len(d), 'persons', g.pid.nunique(), 'multi-firm persons', int((g.groupby('pid').src_corp_code.nunique() > 1).sum()), flush=True)
    return out

def interlock_edges():
    """各年的直接兼任边（两家公司在 FY t 共有至少一名登记董事；严格 as-of 规则）。
    输出：dict 年份 → DataFrame(src, dst, inside)，inside = 共有的人中是否有执行人员（定义同 il_inside）。缓存到 out_dev/interlock_edges.csv。"""
    f = O / 'interlock_edges.csv'
    if not f.exists():
        rows = []
        for t, Pt in persons().items():
            firms_of = collections.defaultdict(dict)
            for c, ps in Pt.items():
                for p, ins in ps.items(): firms_of[p][c] = ins
            ed = {}
            for fs in firms_of.values():
                cs = sorted(fs)
                for i in range(len(cs)):
                    for j in range(i + 1, len(cs)):
                        k = (cs[i], cs[j]); ed[k] = ed.get(k, False) or fs[cs[i]] or fs[cs[j]]
            rows += [(t, a, b, int(ins)) for (a, b), ins in ed.items()]
        pd.DataFrame(rows, columns=['fy', 'src', 'dst', 'inside']).to_csv(f, index=False)
    E = pd.read_csv(f, dtype={'src': str, 'dst': str})
    return {t: g[['src', 'dst', 'inside']].reset_index(drop=True) for t, g in E.groupby('fy')}

def main():
    X = pd.read_csv(D / 'pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str}); X = X[X.fy.isin(YEARS)]
    PS = persons(); rows = []
    for t, g in X.groupby('fy'):
        Pt = PS.get(t, {})
        firms_of = collections.defaultdict(set)
        for c, ps in Pt.items():
            for p in ps: firms_of[p].add(c)
        nb = collections.defaultdict(set)                     # 当年兼任网络：公司 → 共享董事的其他公司
        for fs in firms_of.values():
            for c in fs: nb[c] |= fs - {c}
        ever = collections.defaultdict(set)                   # s ≤ t 任过职的人
        for s in range(2019, t + 1):
            for c, ps in PS.get(s, {}).items(): ever[c] |= set(ps)
        for i, a, o in zip(g.index, g.anchor, g.other):
            pa, po = Pt.get(a, {}), Pt.get(o, {}); sh = pa.keys() & po.keys()
            rows.append(pd.Series(dict(il_direct=np.log1p(len(sh)), il_inside=np.log1p(sum(pa[p] or po[p] for p in sh)),
                                       il_2hop=np.log1p(len((nb[a] & nb[o]) - {a, o})), il_hist=np.log1p(len((ever[a] & ever[o]) - sh)),
                                       il_has_a=float(bool(pa)), il_has_o=float(bool(po))), name=i))
        print(t, len(g), flush=True)
    F = pd.DataFrame(rows)[IL]; F.to_csv(O / ('interlock_feats_lenient.csv' if '--lenient' in sys.argv else 'interlock_feats.csv'), index_label='row')
    J = X.join(F); B = (J[IL[:4]] > 0).assign(label=J.label)
    print('share of pairs with feature > 0, by label:'); print(B.groupby('label').mean().T.round(4).to_string())
    print('coverage', F[['il_has_a', 'il_has_o']].mean().round(3).to_dict())
    deg = {t: collections.Counter(list(pd.read_csv(D / f'graph_{t}.csv', dtype=str).src) + list(pd.read_csv(D / f'graph_{t}.csv', dtype=str).dst)) for t in YEARS}
    Pz = J[J.label == 1].copy(); Pz['iso'] = [min(deg[t][a], deg[t][o]) == 0 for t, a, o in zip(Pz.fy, Pz.anchor, Pz.other)]
    print('positives with any interlock signal, isolated vs not:'); print(Pz.groupby('iso')[IL[:4]].apply(lambda d: (d > 0).mean()).round(3).to_string())

if __name__ == '__main__': main()
