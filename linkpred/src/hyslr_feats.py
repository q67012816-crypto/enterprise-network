"""第 4 项：最大股东及特殊关系人（최대주주현황）点对特征。只算 fy 2019–2023（训练年和验证年；不计算测试年 fy 2024）。
数据：data/interim/hyslr_panel_raw.csv（src/exec_panel.py --endpoint hyslrSttus 下载的年报"최대주주 및 특수관계인의 주식소유 현황"）。
as-of 规则同第 3 项：FY t 只用 rcept_no 日期 ≤ C_t（t+1 年 4 月 30 日）的年报（--lenient 时保留之后的更正年报）。
股东名称：法人用 paper_data._key 归一化（去公司形态词、空白、小写）；自然人只有姓名、没有出生年月，所以同名不同人无法区分。
为降低同名误配：某年在超过 HUB 家公司名单中出现的名称不参与匹配（大多是常见人名或"자사주""우리사주조합"等通用条目）。
特征（a = 投资方，o = 候选企业）：
  hs_same_max   双方"최대주주 본인"是同一主体（同一控制人，通常意味着同一集团）
  hs_shared     双方名单（最大股东 + 特殊关系人）共有的名称数（log1p）
  hs_corp       其中法人名称的个数（log1p；法人名称几乎不会误配）
  hs_exec_ao    o 的股东名单中、同时是 a 的董事或高管（第 3 项임원현황，同年、同名）的人数（log1p）；hs_exec_oa 反之
  hs_has_a / hs_has_o   该公司 FY t 是否有可用的최대주주현황
输出：out_dev/hyslr_feats.csv（或 hyslr_feats_lenient.csv），索引与 pairs.csv 行号一致。用法：python src/hyslr_feats.py [--lenient]"""
import collections, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import _key

L = Path(__file__).resolve().parents[1]; P0 = L.parent; D = L / 'data_v3x'; O = L / 'out_dev'
YEARS = range(2019, 2024); HUB = 5; LEN = '--lenient' in sys.argv
HS = ['hs_same_max', 'hs_shared', 'hs_corp', 'hs_exec_ao', 'hs_exec_oa', 'hs_has_a', 'hs_has_o']
_corp = re.compile(r'\(주\)|㈜|주식회사|\(유\)|유한회사|재단|조합|공사|은행|증권|보험|자산운용|투자|홀딩스|co\.|ltd|inc|corp|llc|fund|펀드', re.I)
_skip = re.compile(r'^(계|합계|소계|총계|-|자기주식|자사주|우리사주조합.*|기타)$')

def asof(df):
    """只保留 API 正常、FY t 年报提交日期 ≤ C_t 的行（--lenient 时不按日期剔除）。"""
    df = df[df.api_status.eq('000')].copy(); df['fy'] = pd.to_numeric(df.bsns_year, errors='coerce')
    if not LEN: df = df[df.rcept_no.str[:8].le((df.fy + 1).astype(int).astype(str) + '0430')]
    return df

def holders():
    """返回 dict 年份 → dict 公司 → {'max': 最大股东本人的键集合, 'all': {键: 是否法人}}，并打印覆盖情况。"""
    H = asof(pd.read_csv(P0 / 'data/interim/hyslr_panel_raw.csv', dtype=str, keep_default_na=False))
    nm = H.nm.str.strip()
    H = H.assign(corp=nm.str.contains(_corp), k=[_key(v) if c else re.sub(r'\s+', '', v) for v, c in zip(nm, nm.str.contains(_corp))],
                 is_max=H.relate.str.replace(r'\s+', '', regex=True).pipe(lambda r: r.str.contains('본인') | r.isin(['최대주주', '최대주주(본인)'])))   # 各公司写法不同：본인 / 최대주주 / 최대주주본인 / 최대주주(본인)
    H = H[H.k.ne('') & ~nm.str.match(_skip)]
    out = {}
    for t, g in H.groupby('fy'):
        n_firms = g.groupby('k').src_corp_code.nunique(); hub = set(n_firms[n_firms > HUB].index)
        d = collections.defaultdict(lambda: {'max': set(), 'all': {}})
        for c, k, corp, mx in zip(g.src_corp_code, g.k, g.corp, g.is_max):
            if k in hub: continue
            d[c]['all'][k] = corp
            if mx: d[c]['max'].add(k)
        out[int(t)] = d
        print(t, 'firms', g.src_corp_code.nunique(), 'names', g.k.nunique(), 'hub names dropped', len(hub),
              'names in ≥2 firms', int(((n_firms >= 2) & (n_firms <= HUB)).sum()), flush=True)
    return out

def exec_names():
    """第 3 项임원현황中每家公司每年的董事和高管姓名（去空白；只用于与股东姓名做同公司内匹配）。"""
    E = asof(pd.read_csv(P0 / 'data/interim/exec_panel_raw.csv', dtype=str, keep_default_na=False))
    out = collections.defaultdict(lambda: collections.defaultdict(set))
    for t, c, n in zip(E.fy, E.src_corp_code, E.nm.str.replace(r'\s+', '', regex=True)):
        if n: out[int(t)][c].add(n)
    return out

def main():
    X = pd.read_csv(D / 'pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str}); X = X[X.fy.isin(YEARS)]
    HD, EX = holders(), exec_names(); rows = []
    for t, g in X.groupby('fy'):
        Ht, Et = HD.get(t, {}), EX.get(t, {})
        for i, a, o in zip(g.index, g.anchor, g.other):
            ha, ho = Ht.get(a), Ht.get(o)
            r = dict(hs_has_a=float(ha is not None), hs_has_o=float(ho is not None))
            if ha is not None and ho is not None:
                sh = ha['all'].keys() & ho['all'].keys()
                r.update(hs_same_max=float(bool(ha['max'] & ho['max'])), hs_shared=np.log1p(len(sh)), hs_corp=np.log1p(sum(ha['all'][k] for k in sh)))
            else: r.update(hs_same_max=0., hs_shared=0., hs_corp=0.)
            r['hs_exec_ao'] = np.log1p(len({k for k, c in (ho or {'all': {}})['all'].items() if not c} & Et.get(a, set())))
            r['hs_exec_oa'] = np.log1p(len({k for k, c in (ha or {'all': {}})['all'].items() if not c} & Et.get(o, set())))
            rows.append(pd.Series(r, name=i))
        print(t, len(g), flush=True)
    F = pd.DataFrame(rows)[HS]; F.to_csv(O / ('hyslr_feats_lenient.csv' if LEN else 'hyslr_feats.csv'), index_label='row')
    J = X.join(F); B = (J[HS[:5]] > 0).assign(label=J.label)
    print('share of pairs with feature > 0, by label:'); print(B.groupby('label').mean().T.round(4).to_string())
    print('coverage', F[['hs_has_a', 'hs_has_o']].mean().round(3).to_dict())
    deg = {t: collections.Counter(list(g_.src) + list(g_.dst)) for t in YEARS for g_ in [pd.read_csv(D / f'graph_{t}.csv', dtype=str)]}
    Pz = J[J.label == 1].copy(); Pz['iso'] = [min(deg[t][a], deg[t][o]) == 0 for t, a, o in zip(Pz.fy, Pz.anchor, Pz.other)]
    print('positives with signal, isolated vs not:'); print(Pz.groupby('iso')[HS[:5]].apply(lambda d: (d > 0).mean()).round(3).to_string())
    IL = pd.read_csv(O / 'interlock_feats.csv', index_col='row').reindex(Pz.index)
    print('positives: any shareholder signal', round(float((Pz[HS[:5]] > 0).any(axis=1).mean()), 3), '| any interlock', round(float((IL[['il_direct', 'il_2hop', 'il_hist']] > 0).any(axis=1).mean()), 3),
          '| either', round(float(((Pz[HS[:5]] > 0).any(axis=1) | (IL[['il_direct', 'il_2hop', 'il_hist']] > 0).any(axis=1)).mean()), 3))

if __name__ == '__main__': main()
