"""开发阶段新增的点对特征（只算 fy 2019–2023，即训练年和验证年；绝不计算测试年 fy 2024 的点对）。
第 1 项 port（投资方完整投资组合，含非上市被投资方）：投资方 FY t 出资表（C_t 前披露、期末持有 > 0）里每个被投资方
  "最近一期财务现황"的总资产和净利润 → 组合规模/盈利/亏损比例，与候选企业比较；投资目的结构（单纯投资/经营参与）× 候选规模。
第 2 项 ind（细行业）：KSIC 前 3/4/5 位是否相同；投资方过去持有过的上市被投资方（G_s，s ≤ t）中与候选同行业的比例，及反向比例。
计算时排除候选企业本身那一行，避免把"已持有"信息带进来。输出缓存到 out_dev/pair_feats.csv（索引与 pairs.csv 行号一致）。
用法：python src/pair_feats.py"""
import collections, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import _key, rd

L = Path(__file__).resolve().parents[1]; P0 = L.parent; D = L / 'data_v3x'; O = L / 'out_dev'
YEARS = range(2019, 2024)
PORT = ['pf_has', 'pf_size_gap', 'pf_size_z', 'pf_prof_gap', 'pf_loss_gap', 'pf_simple_x_size', 'pf_simple_x_Y', 'pf_mgmt_x_size', 'pf_n_x_size']
IND = ['same3', 'same4', 'same5', 'hist_has', 'aff2', 'aff3', 'aff4', 'rev_has', 'rev2', 'rev3']

def num(c): return pd.to_numeric(c.str.replace(',', '', regex=False).str.strip(), errors='coerce')

def portfolio_rows():
    """出资表中可用的行：API 正常、期末持有 > 0、有披露日；附被投资方匹配键、总资产、净利润、投资目的。"""
    V = rd(P0 / 'data/interim/invest_panel_raw.csv'); A = rd(P0 / 'data/interim/filings_A001.csv')
    disc = A.drop_duplicates('rcept_no').set_index('rcept_no').rcept_dt
    pct, q, b = num(V.trmend_blce_qota_rt), num(V.trmend_blce_qy), num(V.trmend_blce_acntbk_amount)
    ok = V.api_status.eq('000') & ~(pct.notna() & ~pct.between(0, 100)) & ((pct.gt(0) & pct.le(100)) | q.gt(0) | b.gt(0))
    V = V[ok].assign(disc=V.rcept_no.map(disc).fillna(''), k=V.inv_prm.map(_key), ta=num(V.recent_bsns_year_fnnr_sttus_tot_assets),
                     ni=num(V.recent_bsns_year_fnnr_sttus_thstrm_ntpf))
    pur = V.invstmnt_purps.str.replace(r'\s+', '', regex=True)
    V['simple'] = pur.str.contains('단순').astype(float); V['mgmt'] = pur.str.contains('경영').astype(float)
    return V[V.k.ne('') & V.disc.ne('')]

def main():
    X = pd.read_csv(D / 'pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str}); X = X[X.fy.isin(YEARS)]
    N = pd.read_csv(D / 'nodes.csv', dtype={'corp_code': str, 'industry': str, 'market': str}, keep_default_na=False, na_values=['']).set_index(['fy', 'corp_code'])
    Ov = rd(P0 / 'data/interim/company_overview.csv'); ind = dict(zip(Ov.corp_code, Ov.induty_code))   # 当前值（与 nodes.industry 同一来源和局限）
    names = collections.defaultdict(set)
    for c, a, s in zip(Ov.corp_code, Ov.corp_name, Ov.stock_name): names[c] |= {_key(a), _key(s)} - {''}
    V = portfolio_rows(); G = {t: rd(D / f'graph_{t}.csv') for t in YEARS}
    tr_la = N.loc[N.index.get_level_values(0).isin([2019, 2020]), 'log_assets'].median()
    same = lambda a, b, k: float(len(a) >= k and len(b) >= k and a[:k] == b[:k])
    out = []
    for t, g in X.groupby('fy'):
        s = V[V.bsns_year.eq(str(t)) & V.disc.le(f'{t + 1}0430')]
        port = {a: d for a, d in s.groupby('src_corp_code')}
        past_out = collections.defaultdict(set); past_in = collections.defaultdict(set)   # 过去持有过的上市被投资方 / 上市投资方
        for u in range(2019, t + 1):
            for a, b in zip(G[u].src, G[u].dst): past_out[a].add(b); past_in[b].add(a)
        for i, a, o in zip(g.index, g.anchor, g.other):
            r = {}
            no = N.loc[(t, o)] if (t, o) in N.index else None
            la = no.log_assets if no is not None and pd.notna(no.log_assets) else np.nan
            nia = no.net_income_assets if no is not None else np.nan
            loss = no.current_operating_loss if no is not None else np.nan
            mkY = float(no is not None and no.market == 'Y')
            d = port.get(a)
            if d is not None: d = d[~d.k.isin(names[o])]   # 排除候选企业本身
            f = d[d.ta > 0] if d is not None else None
            if f is not None and len(f):
                lt = np.log(f.ta.values); m, sd = lt.mean(), lt.std()
                pr = np.clip(f.ni.values / f.ta.values, -1, 1); pr = pr[~np.isnan(pr)]
                ls = float((f.ni < 0).mean())
                r.update(pf_has=1., pf_size_gap=abs(la - m), pf_size_z=(la - m) / (sd + .5),
                         pf_prof_gap=abs(nia - np.median(pr)) if len(pr) else np.nan, pf_loss_gap=abs(loss - ls))
            else: r.update(pf_has=0.)
            if d is not None and len(d):
                c = la - tr_la
                r.update(pf_simple_x_size=d.simple.mean() * c, pf_simple_x_Y=d.simple.mean() * mkY, pf_mgmt_x_size=d.mgmt.mean() * c,
                         pf_n_x_size=np.log1p(d.k.nunique()) * c)
            ia, io = ind.get(a, ''), ind.get(o, '')
            r.update(same3=same(ia, io, 3), same4=same(ia, io, 4), same5=same(ia, io, 5))
            h = [ind.get(z, '') for z in past_out[a] - {o}]
            r['hist_has'] = float(bool(h))
            for k in (2, 3, 4): r[f'aff{k}'] = np.mean([same(z, io, k) for z in h]) if h else 0.
            h = [ind.get(z, '') for z in past_in[o] - {a}]
            r['rev_has'] = float(bool(h))
            for k in (2, 3): r[f'rev{k}'] = np.mean([same(z, ia, k) for z in h]) if h else 0.
            out.append(pd.Series(r, name=i))
        print(t, len(g), flush=True)
    F = pd.DataFrame(out)[PORT + IND]
    F.to_csv(O / 'pair_feats.csv', index_label='row')
    P = X.join(F); print(P.groupby('label')[PORT + IND].mean().T.round(3).to_string()); print('missing share', F.isna().mean().round(3).to_dict())

if __name__ == '__main__': main()
