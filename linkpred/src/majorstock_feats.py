"""探索性（冻结前追加，2026-11-06）：5% 以上大量持股人（OpenDART majorstock，"주식등의 대량보유상황보고"）。
只算 fy 2019–2023（训练年和验证年；不计算测试年）。数据：data/interim/majorstock_raw.csv（src/exec_panel.py --endpoint majorstock；
该接口按发行公司返回全部报告，不分年份）。
as-of 规则：FY t 只用 rcept_dt ≤ C_t（t+1 年 4 月 30 日）的报告；每个（发行公司, 报告人）取 C_t 前最近一次报告的持股比例 stkrt，
≥ 5% 才算持有。共同持股人的报告以“代表报告人”名义提交，所以这里的报告人是代表报告人（限制）。
报告人名称：用 paper_data._key 归一化（去公司形态词、空白、小写）。
剔除：
  - 报告人是合格上市公司本身（其持股已部分体现在持股图中，属于结构信息）；
  - 报告人就是发行公司本身；
  - 某年持有超过 HUB = 20 家公司 5% 以上股份的报告人（如国民年金），规则同共同投资图的 hub 上限。
机构 / 个人：按名称关键词区分（法人形态词、运用、投资、基金、年金、银行、证券、保险、资本、合伙、外文公司词等为机构，其余为个人）。
特征（a = 投资方，o = 候选企业）：
  ms_common         FY t 同时持有 a 和 o 5% 以上股份的第三方数量（log1p）
  ms_common_inst    其中机构的数量（log1p）
  ms_common_indiv   其中个人的数量（log1p）
  ms_has_a / ms_has_o   该公司 FY t 是否有 C_t 前的 5% 报告（覆盖指示）
输出：out_dev/majorstock_feats.csv（索引与 pairs.csv 行号一致）。用法：python src/majorstock_feats.py"""
import collections, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import _key, rd

L = Path(__file__).resolve().parents[1]; P0 = L.parent; D = L / 'data_v3x'; O = L / 'out_dev'
YEARS = range(2019, 2024); HUB = 20
MS = ['ms_common', 'ms_common_inst', 'ms_common_indiv', 'ms_has_a', 'ms_has_o']
_inst = re.compile(r'\(주\)|㈜|주식회사|\(유\)|유한|합자|합명|재단|조합|공사|공단|연금|은행|증권|보험|생명|화재|캐피탈|자산운용|운용|투자|파트너스|홀딩스|인베스트|'
                   r'펀드|기금|사모|신탁|그룹|컴퍼니|코리아|글로벌|co\.|ltd|inc|corp|llc|l\.p|lp\b|limited|fund|capital|partners|management|bank|trust|holdings|gmbh|s\.a', re.I)

def holders():
    """返回 dict 年份 → dict 发行公司 → {报告人键: 是否机构}（as-of、≥5%、剔除规则之后），并打印覆盖情况。"""
    M = pd.read_csv(P0 / 'data/interim/majorstock_raw.csv', dtype=str, keep_default_na=False)
    M = M[M.api_status.eq('000') & M.repror.str.strip().ne('')].copy()
    M['rt'] = pd.to_numeric(M.stkrt.str.replace(',', '', regex=False), errors='coerce'); M['dt'] = M.rcept_dt.str.replace('-', '', regex=False)
    M['k'] = M.repror.map(_key); M['inst'] = M.repror.str.contains(_inst)
    U = set(rd(P0 / 'data/processed/universe_firms.csv').corp_code); Ov = rd(P0 / 'data/interim/company_overview.csv')
    listed = {_key(n) for c, a, b in zip(Ov.corp_code, Ov.corp_name, Ov.stock_name) if c in U for n in (a, b)} - {''}
    own = {c: {_key(a), _key(b)} for c, a, b in zip(Ov.corp_code, Ov.corp_name, Ov.stock_name)}
    M = M[M.k.ne('') & ~M.k.isin(listed)]
    out = {}
    for t in YEARS:
        s = M[M.dt.le(f'{t + 1}0430')].sort_values(['dt', 'rcept_no']).groupby(['src_corp_code', 'k']).tail(1)   # C_t 前最近一次报告
        s = s[(s.rt >= 5) & [k not in own.get(c, set()) for c, k in zip(s.src_corp_code, s.k)]]
        n_firms = s.groupby('k').src_corp_code.nunique(); hub = set(n_firms[n_firms > HUB].index)
        d = collections.defaultdict(dict)
        for c, k, ins in zip(s.src_corp_code, s.k, s.inst):
            if k not in hub: d[c][k] = bool(ins)
        reported = set(M[M.dt.le(f'{t + 1}0430')].src_corp_code)
        out[t] = (d, reported)
        print(t, 'issuers with ≥5% outside holder', len(d), '| issuers with any report by C_t', len(reported), '| holders', s.k.nunique(), '| hub holders dropped', len(hub),
              '| holders in ≥2 issuers', int(((n_firms >= 2) & (n_firms <= HUB)).sum()), flush=True)
    return out

def main():
    X = pd.read_csv(D / 'pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str}); X = X[X.fy.isin(YEARS)]
    HD = holders(); rows = []
    for t, g in X.groupby('fy'):
        d, rep = HD[t]
        for i, a, o in zip(g.index, g.anchor, g.other):
            ha, ho = d.get(a, {}), d.get(o, {}); sh = ha.keys() & ho.keys(); ni = sum(ha[k] for k in sh)
            rows.append(pd.Series(dict(ms_common=np.log1p(len(sh)), ms_common_inst=np.log1p(ni), ms_common_indiv=np.log1p(len(sh) - ni),
                                       ms_has_a=float(a in rep), ms_has_o=float(o in rep)), name=i))
        print(t, len(g), flush=True)
    F = pd.DataFrame(rows)[MS]; F.to_csv(O / 'majorstock_feats.csv', index_label='row')
    J = X.join(F); P = J.label == 1
    print('share of pairs with feature > 0, by label:'); print((J[MS[:3]] > 0).assign(label=J.label).groupby('label').mean().T.round(4).to_string())
    print('coverage', F[['ms_has_a', 'ms_has_o']].mean().round(3).to_dict())
    IL = pd.read_csv(O / 'interlock_feats.csv', index_col='row').reindex(J.index); HSf = pd.read_csv(O / 'hyslr_feats.csv', index_col='row').reindex(J.index)
    ms_any = J.ms_common > 0; il_any = (IL[['il_direct', 'il_2hop', 'il_hist']] > 0).any(axis=1)
    hs_any = (HSf[['hs_same_max', 'hs_shared', 'hs_corp', 'hs_exec_ao', 'hs_exec_oa']] > 0).any(axis=1)
    print('positives: major-holder', round(float(ms_any[P].mean()), 4), '| interlock', round(float(il_any[P].mean()), 4), '| shareholder(item 4)', round(float(hs_any[P].mean()), 4),
          '| ms & interlock', round(float((ms_any & il_any)[P].mean()), 4), '| ms & item4', round(float((ms_any & hs_any)[P].mean()), 4),
          '| ms without interlock or item4', round(float((ms_any & ~il_any & ~hs_any)[P].mean()), 4))
    deg = {t: collections.Counter(list(g_.src) + list(g_.dst)) for t in YEARS for g_ in [pd.read_csv(D / f'graph_{t}.csv', dtype=str)]}
    iso = pd.Series([min(deg[t][a], deg[t][o]) == 0 for t, a, o in zip(J.fy, J.anchor, J.other)], index=J.index)
    print('positives with major-holder signal: isolated', round(float(ms_any[P & iso].mean()), 4), '| connected', round(float(ms_any[P & ~iso].mean()), 4))

if __name__ == '__main__': main()
