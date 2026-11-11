"""命题 4（paper/THEORY_DRAFT.md）的定量检验：点对信号带来的平均组内 AUC 增益上界。只读三个验证年的已有分数，不碰测试年。
设基础分数 b，oracle 分数 b' = b + M·s（先按 s 排序，再按 b 排序；b 是概率，取 M = 2 即可），s 为点对的兼任信号（0/1）。
上界 = (1/n) Σ_{p ∈ S⁺} (1 − AUC_p)，S⁺ = 正例点对 s = 1 的正例，n 为该组正例数。
对每组（全部 / 低连接 ≤1 / 孤立 0）报告：(a) S⁺ 占比；(b) 负例中 s = 1 的占比；(c) 上界；(d) 按 b' 计算的 oracle 增益；
(e) 实际学到的增益（M1 + il − M1；DFGT + port + fine + il − DFGT + port + fine，10 种子平均分数）。
s 两种定义：il_direct > 0；任一兼任特征（il_direct、il_inside、il_2hop、il_hist）> 0。
用法：python src/prop4.py   → 输出 out_dev_gpu10/prop4.json"""
import collections, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import rd, within_metrics

L = Path(__file__).resolve().parents[1]; D = L / 'data_v3x'
BASES = {'M1': (L / 'out_dev' / 'scores_M1.csv', L / 'out_dev' / 'scores_M1+il.csv'),
         'dfgt+port+fine': (L / 'out_dev_gpu10' / 'scores_dfgt+port+fine.csv', L / 'out_dev_gpu10' / 'scores_dfgt+port+fine+il.csv')}

def main():
    IL = pd.read_csv(L / 'out_dev' / 'interlock_feats.csv', index_col='row')
    X = pd.read_csv(D / 'pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str})
    sig = {'il_direct>0': (IL.il_direct > 0), 'any_il>0': (IL[['il_direct', 'il_inside', 'il_2hop', 'il_hist']] > 0).any(axis=1)}
    deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t in range(2019, 2025) for g in [rd(D / f'graph_{t}.csv')]}
    out = {}
    for bname, (fb, fa) in BASES.items():
        S = pd.read_csv(fb, dtype={'anchor': str, 'other': str, 'pair_id': str}); A = pd.read_csv(fa, dtype={'anchor': str, 'other': str, 'pair_id': str})
        key = X.reset_index()[['index', 'pair_id', 'anchor', 'other']].rename(columns={'index': 'row'})
        S = S.merge(key, on=['pair_id', 'anchor', 'other'], how='left'); assert S.row.notna().all()
        A = A.merge(key, on=['pair_id', 'anchor', 'other'], how='left')
        S['pos_group'] = ''; A['pos_group'] = ''
        W0 = within_metrics(S, 'score')[['AUC']]
        Wl = within_metrics(A, 'score')[['AUC']].rename(columns={'AUC': 'AUC_learned'})
        pos = S[S.label == 1].set_index('pair_id')
        md = pd.Series([min(deg[t][a], deg[t][o]) for t, a, o in zip(pos.fy, pos.anchor, pos.other)], index=pos.index)
        grp = {'all': md >= 0, 'low': md <= 1, 'iso': md == 0}
        for sname, sv in sig.items():
            s_pair = sv.reindex(S.row.astype(int)).fillna(False).astype(float).values
            S2 = S.assign(sc=S.score + 2 * s_pair); W1 = within_metrics(S2, 'sc')[['AUC']].rename(columns={'AUC': 'AUC_oracle'})
            s_pos = pd.Series(s_pair[(S.label == 1).values], index=pos.index)
            neg = S[S.label == 0]; neg_s = pd.Series(s_pair[(S.label == 0).values], index=neg.index)
            neg_g = neg.pair_id.map(md)   # 负例按所属正例的组归类
            W = W0.join(W1).join(Wl).join(s_pos.rename('s'))
            for g, m in grp.items():
                w = W.loc[m[m].index]; nm = neg_g.reindex(neg.index)
                nmask = (nm <= 1) if g == 'low' else (nm == 0) if g == 'iso' else nm >= 0
                r = dict(n=int(len(w)), S_plus_share=round(float(w.s.mean()), 4), neg_signal_share=round(float(neg_s[nmask.values].mean()), 5),
                         base_auc=round(float(w.AUC.mean()), 4), upper_bound=round(float(((1 - w.AUC) * w.s).mean()), 4),
                         oracle_gain=round(float((w.AUC_oracle - w.AUC).mean()), 4), learned_gain=round(float((w.AUC_learned - w.AUC).mean()), 4))
                out.setdefault(bname, {}).setdefault(sname, {})[g] = r
                print(f'{bname:15s} {sname:12s} {g:4s} n={r["n"]:3d}  S+ {r["S_plus_share"]:.3f}  neg s=1 {r["neg_signal_share"]:.5f}  base {r["base_auc"]:.3f}  '
                      f'bound {r["upper_bound"]:+.4f}  oracle {r["oracle_gain"]:+.4f}  learned {r["learned_gain"]:+.4f}', flush=True)
    json.dump(out, open(L / 'out_dev_gpu10' / 'prop4.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

if __name__ == '__main__': main()
