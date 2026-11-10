"""DEV_RESULTS 第 28 节：全候选集重评分（只用逻辑回归；冻结代码不改，按 backcast 的方式在进程内改常量）。
用法（按顺序）：
  python src/fullcand.py build    # data_v3f/pairs.csv（原行 + 三个验证年的全候选扩展行）+ 复制 graphs/nodes/eligible；核对原 100 负例 ⊂ 全候选
  python src/fullcand.py feats    # 冻结 interlock_feats 对 data_v3f 全部行计算 → out_fullcand/feat/
  python src/fullcand.py score    # 每折训练（只用原始训练行）→ 给原验证行 + 扩展行打分；回归检查：原 100 负例子集分数与已存分数一致
  python src/fullcand.py eval     # 主检验与报告 → out_fullcand/fullcand_eval.json"""
import collections, contextlib, io, json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
L = Path(__file__).resolve().parents[1]; DX = L / 'data_v3x'; DF = L / 'data_v3f'; OF = L / 'out_fullcand'; FT = OF / 'feat'
cmd = sys.argv[1]
VAL = {2021: 'F1', 2022: 'F2', 2023: 'F3'}; FOLDS = {'F1': ((2019, 2020), 2021), 'F2': ((2019, 2020, 2021), 2022), 'F3': ((2019, 2020, 2021, 2022), 2023)}
rd = lambda p, **k: pd.read_csv(p, dtype=str, keep_default_na=False, **k)

if cmd == 'build':
    if (DF / 'pairs.csv').exists(): sys.exit('data_v3f 已存在，拒绝覆盖')
    DF.mkdir()
    X = pd.read_csv(DX / 'pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str})
    el = rd(DX / 'eligible.csv'); elig = {int(t): set(g.corp_code) for t, g in el.groupby('fy')}
    H = rd(DX / 'history_undirected.csv'); Ht = {int(t): {frozenset((a, b)) for a, b in zip(g.a, g.b)} for t, g in H.groupby('fy')}
    G1 = {t: rd(DX / f'graph_{t + 1}.csv') for t in VAL}
    rows = []; chk_ok = chk_bad = 0
    for t in VAL:
        E = elig[t] & elig[t + 1]; Eg = sorted(E)
        blocked = Ht[t] | {frozenset((a, b)) for a, b in zip(G1[t].src, G1[t].dst)}
        pos = X[(X.fy == t) & (X.label == 1)]
        orig_neg = X[(X.fy == t) & (X.label == 0)].groupby('pair_id').other.apply(set)
        for pid, u in zip(pos.pair_id, pos.anchor):
            cand = [w for w in Eg if w != u and frozenset((u, w)) not in blocked]
            if orig_neg[pid] <= set(cand): chk_ok += 1
            else: chk_bad += 1
            rows += [dict(pair_id=pid, fy=t, split='fullcand', anchor=u, other=w, label=0, pct=np.nan, directions='', new_evidence='') for w in cand]
    assert chk_bad == 0, f'原负例不是全候选子集：{chk_bad} 个正例'
    EX = pd.DataFrame(rows)
    for c in X.columns:
        if c not in EX.columns: EX[c] = X[c].iloc[0] if c == 'also_in_2024_doc_set' else False
    U = pd.concat([X, EX[X.columns]], ignore_index=True)
    U.to_csv(DF / 'pairs.csv', index=False)
    for f in ['nodes.csv', 'eligible.csv'] + [f'graph_{t}.csv' for t in range(2019, 2026)]:
        if (DX / f).exists(): shutil.copyfile(DX / f, DF / f)
    print('rows: orig', len(X), 'extended', len(EX), 'union', len(U), '| subset check ok', chk_ok)
    print('cand per positive:', EX.groupby('pair_id').size().describe().round(1).to_dict())
elif cmd == 'feats':
    if (FT / 'interlock_feats.csv').exists(): sys.exit('特征已存在，拒绝覆盖')
    FT.mkdir(parents=True, exist_ok=True)
    import interlock_feats as IF
    IF.D = DF; IF.O = FT; IF.YEARS = range(2019, 2024)
    with contextlib.redirect_stdout(io.StringIO()): IF.main()
    n = pd.read_csv(DF / 'pairs.csv', usecols=['fy'])
    n[n.fy.between(2019, 2023)].iloc[:, :0].to_csv(FT / 'pair_feats.csv', index_label='row')
    print('feats written')
elif cmd == 'score':
    argv0 = sys.argv; sys.argv = ['paper_dev.py', 'none', '--out', 'out_fullcand', '--device', 'cpu']
    import paper_dev as PD
    sys.argv = argv0; PD.D = DF; PD.FE = FT
    X, N, G = PD.load(); cols = ['pair_id', 'fy', 'anchor', 'other', 'label', 'split']
    IL = pd.read_csv(FT / 'interlock_feats.csv', index_col='row')
    for v in ('M1', 'M1+il'):
        if (OF / f'scores_{v}.csv').exists(): print('skip', v, '(exists)'); continue   # 变体级续跑；LR 确定性，重跑结果相同
        model, opt = PD.VARIANTS[v]; out = []
        for fold, (tr, va) in FOLDS.items():
            sl = X[(X.fy.isin(tr) & (X.split != 'fullcand')) | (X.fy == va)]
            lbl = sl.index.values; Xf = sl.reset_index(drop=True)   # 冻结代码按位置索引特征矩阵，所以每折把行号重排为 0..n-1，特征文件同步重排
            FTf = OF / f'feat_{fold}'; FTf.mkdir(exist_ok=True)
            ILf = IL.loc[lbl].reset_index(drop=True); ILf.index.name = 'row'; ILf.to_csv(FTf / 'interlock_feats.csv')
            Xf.iloc[:, :0].to_csv(FTf / 'pair_feats.csv', index_label='row')
            PD.FE = FTf
            Ys = PD.node_tables(N, G, list(tr) + [va], tr, False)
            s = PD.run_lr(Xf, Ys, tr, va, opt)
            out.append(Xf[Xf.fy == va][cols].assign(fold=fold, score=s))
        pd.concat(out).to_csv(OF / f'scores_{v}.csv', index=False)
elif cmd == 'regress':
    for v in ('M1', 'M1+il'):
        S = pd.read_csv(OF / f'scores_{v}.csv', dtype={'pair_id': str, 'anchor': str, 'other': str})
        S0 = S[S.split != 'fullcand']
        R = pd.read_csv(L / 'out_dev_rewire/real' / f'scores_{v}.csv', dtype={'pair_id': str, 'anchor': str, 'other': str})
        a = S0.sort_values(['pair_id', 'other']).score.values; b = R.sort_values(['pair_id', 'other']).score.values
        print(v, 'regression max|diff| on original rows:', float(np.abs(a - b).max()) if len(a) == len(b) else f'LEN {len(a)} vs {len(b)}')
elif cmd == 'eval':
    deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t in VAL for g in [rd(DX / f'graph_{t}.csv')]}
    il = pd.read_csv(FT / 'interlock_feats.csv', index_col='row')
    sig = (il[['il_direct', 'il_inside', 'il_2hop', 'il_hist']] > 0).any(axis=1)
    XP = pd.read_csv(DF / 'pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str})
    sig = sig.reindex(XP.index).fillna(False)
    def per_pos(v, full):
        S = pd.read_csv(OF / f'scores_{v}.csv', dtype={'pair_id': str, 'anchor': str, 'other': str})
        r = {}
        for pid, g in S.groupby('pair_id'):
            p = g[g.label == 1].iloc[0]
            n = g if full else g[g.split != 'fullcand']
            n = n[n.label == 0]
            s = p.score; rk = 1 + (n.score.values > s).sum() + .5 * (n.score.values == s).sum()
            r[pid] = (p.fold, p.anchor, int(p.fy), min(deg[p.fy][p.anchor], deg[p.fy][p.other]),
                      ((n.score.values < s).sum() + .5 * (n.score.values == s).sum()) / len(n), rk, len(n))
        return pd.DataFrame(r, index=['fold', 'anchor', 'fy', 'md', 'auc', 'rank', 'ncand']).T
    P = {(v, f): per_pos(v, f) for v in ('M1', 'M1+il') for f in (True, False)}
    W = P[('M1', True)][['fold', 'anchor', 'fy', 'md', 'ncand']].copy()
    W['md'] = W.md.astype(int); W['fy'] = W.fy.astype(int); W['cl'] = W.fold + '-' + W.anchor
    for (v, f), p in P.items():
        tag = f'{v}|{"full" if f else "u100"}'
        W[f'{tag}|auc'] = p.auc.reindex(W.index).astype(float)
        r = p['rank'].reindex(W.index).astype(float)
        W[f'{tag}|H1'] = (r <= 1).astype(float); W[f'{tag}|H10'] = (r <= 10).astype(float); W[f'{tag}|rr'] = 1 / r; W[f'{tag}|rank'] = r
    # 每个正例面对的带兼任信号负例个数（全候选）
    ext = XP[XP.split == 'fullcand']; nsig = ext.groupby('pair_id').apply(lambda g: int(sig.loc[g.index].sum()))
    W['nsig'] = nsig.reindex(W.index).fillna(0).astype(int)
    cl = W.cl.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}
    rng = np.random.default_rng(20261103); DR = [np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]) for _ in range(10000)]
    def ci(d, m):
        v = [d[i][m[i]].mean() for i in DR if m[i].any()]
        return [round(float(d[m].mean()), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4), round(float(np.percentile(v, 5)), 4), int(m.sum())]
    GR = {'low': (W.md <= 1).values, 'iso': (W.md == 0).values, 'd1': (W.md == 1).values, 'high': (W.md >= 2).values, 'all': np.ones(len(W), bool)}
    gain_full = (W['M1+il|full|auc'] - W['M1|full|auc']).values; gain_u = (W['M1+il|u100|auc'] - W['M1|u100|auc']).values
    res = dict(ncand=W.ncand.astype(float).describe().round(1).to_dict(),
               nsig_per_pos=dict(mean=round(float(W.nsig.mean()), 2), median=int(W.nsig.median()), max=int(W.nsig.max()), zero_share=round(float((W.nsig == 0).mean()), 3)),
               primary_full={g: ci(gain_full, m) for g, m in GR.items()},
               u100_same_scores={g: ci(gain_u, m) for g, m in GR.items()},
               per_year_low={t: ci(gain_full, GR['low'] & (W.fy == t).values) for t in VAL},
               levels={g: {f'{v}|{f}': round(float(W.loc[m, f'{v}|{f}|auc'].mean()), 4) for v in ('M1', 'M1+il') for f in ('full', 'u100')} for g, m in GR.items()},
               ranking_full={g: {mm: ci((W[f'M1+il|full|{mm}'] - W[f'M1|full|{mm}']).values, m) for mm in ('H1', 'H10', 'rr')} for g, m in (('low', GR['low']), ('all', GR['all']))},
               median_rank={g: {f'{v}|{f}': float(W.loc[m, f'{v}|{f}|rank'].median()) for v in ('M1', 'M1+il') for f in ('full', 'u100')} for g, m in GR.items()})
    lo = res['primary_full']['low']
    res['verdict'] = '全候选集下保持' if lo[3] > 0 else ('方向一致但不显著' if lo[0] > 0 else '不保持')
    res['ratio_low_full_vs_u100'] = round(lo[0] / res['u100_same_scores']['low'][0], 3) if res['u100_same_scores']['low'][0] else None
    json.dump(res, open(OF / 'fullcand_eval.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False, indent=1))
