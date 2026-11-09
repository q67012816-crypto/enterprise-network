"""DEV_RESULTS 第 25.1–25.3 节（只用开发年；只用逻辑回归；不改冻结代码）。
用法：python src/round3.py run    # 拟合（写 out_dev_rewire_nohist/、out_dev_shproxy/）
      python src/round3.py eval   # 写 out_dev_rewire/round3_eval.json"""
import collections, contextlib, io, json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
L = Path(__file__).resolve().parents[1]; RW = L / 'out_dev_rewire'; NH = L / 'out_dev_rewire_nohist'; SH = L / 'out_dev_shproxy'
argv0 = sys.argv; sys.argv = ['paper_dev.py', 'none', '--out', 'out_dev_rewire', '--device', 'cpu']
import paper_dev as PD
sys.argv = argv0; cmd = argv0[1]

def fit(d, variants):
    PD.FE = d; PD.O = d
    with contextlib.redirect_stdout(io.StringIO()):
        for v in variants: PD.run(v)

if cmd == 'run':
    # 25.1：il_hist 置 0
    for src in sorted([RW / 'real'] + list(RW.glob('N1_*')) + list(RW.glob('N2_*'))):
        d = NH / src.name; d.mkdir(parents=True, exist_ok=True)
        if (d / 'scores_M1+il.csv').exists(): continue
        F = pd.read_csv(src / 'interlock_feats.csv', index_col='row'); F['il_hist'] = 0.; F.to_csv(d / 'interlock_feats.csv', index_label='row')
        for f in ('pair_feats.csv', 'hyslr_feats.csv'): shutil.copyfile(src / f, d / f)
        fit(d, ['M1+il']); print('nohist', src.name, flush=True)
    # 25.3：共享股东 S（放在 ms 组位置）
    SH.mkdir(exist_ok=True)
    for f in ('pair_feats.csv', 'hyslr_feats.csv', 'interlock_feats.csv'): shutil.copyfile(RW / 'real' / f, SH / f)
    H = pd.read_csv(L / 'out_dev' / 'hyslr_feats.csv', index_col='row'); cov = (H.hs_has_a == 1) & (H.hs_has_o == 1)
    S = (((H.hs_same_max > 0) | (H.hs_corp > 0) | (H.hs_shared > 0)) & cov).astype(float)
    pd.DataFrame({'ms_common': S, 'ms_common_inst': H.hs_has_a, 'ms_common_indiv': H.hs_has_o, 'ms_has_a': 0., 'ms_has_o': 0.}, index=H.index).to_csv(SH / 'majorstock_feats.csv', index_label='row')
    pd.DataFrame({'S': S, 'cov': cov.astype(int)}, index=H.index).to_csv(SH / 'shproxy.csv', index_label='row')
    fit(SH, ['M1+ms', 'M1+il+ms']); print('shproxy fitted')
else:
    deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t in (2021, 2022, 2023) for g in [pd.read_csv(L / f'data_v3x/graph_{t}.csv', dtype=str)]}
    def per_pos(f):
        S = pd.read_csv(f, dtype={'pair_id': str, 'anchor': str, 'other': str}); r = {}
        for pid, g in S.groupby('pair_id'):
            p = g[g.label == 1].iloc[0]; n = g.score.values[g.label.values == 0]; s = p.score
            r[pid] = (p.fold, p.anchor, p.fy, min(deg[p.fy][p.anchor], deg[p.fy][p.other]), ((n < s).sum() + .5 * (n == s).sum()) / len(n))
        return pd.DataFrame(r, index=['fold', 'anchor', 'fy', 'md', 'auc']).T
    W = per_pos(RW / 'real' / 'scores_M1.csv'); W['md'] = W.md.astype(int); W['fy'] = W.fy.astype(int); W['cl'] = W.fold + '-' + W.anchor; W['M1'] = W.auc.astype(float)
    W['M1+il'] = per_pos(RW / 'real' / 'scores_M1+il.csv').auc.astype(float)
    def ci(d, m, U=None, B=1000):
        U = W if U is None else U; cl = U.cl.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(20261103)
        v = [];
        for _ in range(B):
            i = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]); mm = m[i]
            if mm.any(): v.append(d[i][mm].mean())
        return [round(float(d[m].mean()), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4), int(m.sum())]
    GR = lambda U: {'low': (U.md <= 1).values, 'iso': (U.md == 0).values, 'all': np.ones(len(U), bool)}
    res = {}
    # 25.1
    real_nh = per_pos(NH / 'real' / 'scores_M1+il.csv').auc.astype(float).reindex(W.index)
    out = dict(real_full_gain={g: ci((W['M1+il'] - W['M1']).values, m) for g, m in GR(W).items()},
               real_nohist_gain={g: ci((real_nh - W['M1']).values, m) for g, m in GR(W).items()},
               hist_contribution={g: ci((W['M1+il'] - real_nh).values, m) for g, m in GR(W).items()}, null={})
    for mdl in ('N1', 'N2'):
        Z = pd.DataFrame({d.name: per_pos(d / 'scores_M1+il.csv').auc.astype(float) for d in sorted(NH.glob(f'{mdl}_*'))}).reindex(W.index); o = {}
        for g, m in GR(W).items():
            Gr = float((real_nh - W['M1']).values[m].mean()); G0 = [float((Z[c] - W['M1']).values[m].mean()) for c in Z]
            o[g] = dict(real=round(Gr, 4), null_mean=round(float(np.mean(G0)), 4), null_sd=round(float(np.std(G0, ddof=1)), 4), null_min=round(min(G0), 4), null_max=round(max(G0), 4),
                        p=round((1 + sum(x >= Gr for x in G0)) / (len(G0) + 1), 4), excess=ci((real_nh - Z.mean(axis=1)).values, m), draws=len(G0))
        out['null'][mdl] = o
    o = out['null']['N2']['low']; out['reading'] = '只看当年共享人员就足以产生增益' if o['p'] <= 1 / 21 + 1e-9 and o['excess'][1] > 0 else ('部分支持' if o['excess'][0] > 0 else '没有证据')
    res['25.1_nohist'] = out
    # 25.2
    py = {}
    for t in (2021, 2022, 2023):
        U = W[W.fy == t]; py[t] = {g: ci((U['M1+il'] - U['M1']).values, m, U) for g, m in GR(U).items()}
    ho = {}
    for f in ('H2', 'H3'):
        A_ = per_pos(L / 'out_dev_holdout' / 'honest' / 'scores_M1.csv'); B_ = per_pos(L / 'out_dev_holdout' / 'honest' / 'scores_M1+il.csv')
        U = A_[A_.fold == f].copy(); U['md'] = U.md.astype(int); U['cl'] = U.fold + '-' + U.anchor; d = (B_.auc.reindex(U.index).astype(float) - U.auc.astype(float)).values
        ho[f] = {g: ci(d, m, U) for g, m in GR(U).items()}
    res['25.2_per_year'] = dict(folds=py, holdout_LR=ho)
    # 25.3
    X = pd.read_csv(L / 'data_v3x' / 'pairs.csv', usecols=['pair_id', 'label'], dtype={'pair_id': str}); sp = pd.read_csv(SH / 'shproxy.csv', index_col='row')
    pos = X[X.label == 1].join(sp).set_index('pair_id').reindex(W.index)
    W['sub'] = np.where(pos['cov'] == 1, np.where(pos.S == 1, 'S=1', 'S=0'), 'S undefined')
    W['ms'] = per_pos(SH / 'scores_M1+ms.csv').auc.astype(float); W['il+ms'] = per_pos(SH / 'scores_M1+il+ms.csv').auc.astype(float)
    res['25.3_shareholder_proxy'] = dict(il_beyond_S={g: ci((W['il+ms'] - W['ms']).values, m) for g, m in GR(W).items()},
                                         S_alone={g: ci((W['ms'] - W['M1']).values, m) for g, m in GR(W).items()},
                                         subsets={s: {g: (ci((W['M1+il'] - W['M1']).values, m & (W['sub'] == s).values) if (m & (W['sub'] == s).values).sum() >= 5 else None) for g, m in GR(W).items()} for s in ('S=1', 'S=0', 'S undefined')},
                                         counts=W.groupby('sub').size().to_dict(), counts_low=W[W.md <= 1].groupby('sub').size().to_dict())
    json.dump(res, open(RW / 'round3_eval.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=str); print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
