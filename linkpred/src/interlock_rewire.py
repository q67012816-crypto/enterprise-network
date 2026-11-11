"""DEV_RESULTS 第 24 节：共同任职的随机重连对照（只用开发年；只用逻辑回归；不改冻结代码）。
  real    用冻结的 interlock_feats.persons() / main() 重算真实特征（须与 out_dev/interlock_feats.csv 逐字节相同），并拟合 M1、M1+il、
          以及公司边际量对照 M1+ms、M1+il+ms（边际量放在 majorstock_feats.csv 的 ms 列位置）。
  null    N1：每年保持双边度数的随机边交换；N2：只在同一 2 位行业的公司之间交换。各 20 次（种子 1–20），每次重算兼任特征并拟合 M1+il。
  eval    写 out_dev_rewire/rewire_eval.json。
用法：python src/interlock_rewire.py real | null --model N1 [--draws 1-20] | eval"""
import argparse, collections, contextlib, io, json, random, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['real', 'null', 'eval']); ap.add_argument('--model', choices=['N1', 'N2']); ap.add_argument('--draws', default='1-20')
A = ap.parse_args()
L = Path(__file__).resolve().parents[1]; P0 = L.parent; R = L / 'out_dev_rewire'; R.mkdir(exist_ok=True)
import interlock_feats as IF
argv0 = sys.argv; sys.argv = ['paper_dev.py', 'none', '--out', 'out_dev_rewire', '--device', 'cpu']
import paper_dev as PD
sys.argv = argv0

def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()): return fn(*a)

def fit(d, variants):
    """在目录 d 中用冻结的 paper_dev.run 拟合；特征从 d 读取，分数写到 d。"""
    PD.FE = d; PD.O = d
    for v in variants: quiet(PD.run, v)

def feats(d, PS):
    """用冻结的 interlock_feats.main() 在目录 d 中按人员数据 PS 计算兼任特征。"""
    d.mkdir(parents=True, exist_ok=True); IF.O = d; orig = IF.persons; IF.persons = lambda: PS
    try: quiet(IF.main)
    finally: IF.persons = orig
    for f in ('pair_feats.csv', 'hyslr_feats.csv'): shutil.copyfile(L / 'out_dev' / f, d / f)

def rewire(PS, seed, strat):
    """每年保持双边度数的随机边交换（10 × 边数次尝试）；执行人员标记跟着人走。strat：公司 → 行业（N2），None 为 N1。"""
    rng = random.Random(seed); out = {}
    for t in sorted(PS):
        E = [(p, f, ins) for f, ps in PS[t].items() for p, ins in ps.items()]; S = {(p, f) for p, f, _ in E}
        pools = collections.defaultdict(list)
        for i, (_, f, _) in enumerate(E): pools[strat.get(f, '') if strat else ''].append(i)
        for _ in range(10 * len(E)):
            i = rng.randrange(len(E)); p1, f1, s1 = E[i]; pool = pools[strat.get(f1, '') if strat else '']
            j = pool[rng.randrange(len(pool))]; p2, f2, s2 = E[j]
            if p1 == p2 or f1 == f2 or (p1, f2) in S or (p2, f1) in S: continue
            S -= {(p1, f1), (p2, f2)}; S |= {(p1, f2), (p2, f1)}; E[i] = (p1, f2, s1); E[j] = (p2, f1, s2)
        d = collections.defaultdict(dict)
        for p, f, ins in E: d[f][p] = ins
        out[t] = d
    return out

if A.cmd == 'real':
    PS = quiet(IF.persons); d = R / 'real'; feats(d, PS)
    same = (d / 'interlock_feats.csv').read_bytes() == (L / 'out_dev' / 'interlock_feats.csv').read_bytes()
    print('real interlock_feats identical to out_dev:', same)
    if not same: sys.exit('真实特征与 out_dev 不一致，停止（第 24 节）')
    # 公司边际量：人数、多重任职人数（同年在 ≥ 2 家公司任职）
    X = pd.read_csv(L / 'data_v3x' / 'pairs.csv', dtype={'anchor': str, 'other': str}); X = X[X.fy.isin(IF.YEARS)]
    size, multi = {}, {}
    for t, Pt in PS.items():
        cnt = collections.Counter(p for ps in Pt.values() for p in ps)
        for f, ps in Pt.items(): size[(t, f)] = len(ps); multi[(t, f)] = sum(cnt[p] >= 2 for p in ps)
    M = pd.DataFrame({'ms_common': [np.log1p(size.get((t, a), 0)) for t, a in zip(X.fy, X.anchor)], 'ms_common_inst': [np.log1p(size.get((t, o), 0)) for t, o in zip(X.fy, X.other)],
                      'ms_common_indiv': [np.log1p(multi.get((t, a), 0)) for t, a in zip(X.fy, X.anchor)], 'ms_has_a': [np.log1p(multi.get((t, o), 0)) for t, o in zip(X.fy, X.other)], 'ms_has_o': 0.}, index=X.index)
    M.to_csv(d / 'majorstock_feats.csv', index_label='row')   # 注意：列名沿用 ms 组的位置，内容是第 24 节的公司边际量
    fit(d, ['M1', 'M1+il', 'M1+ms', 'M1+il+ms']); print('real fitted')
elif A.cmd == 'null':
    PS = quiet(IF.persons); strat = None
    if A.model == 'N2':
        O_ = pd.read_csv(P0 / 'data/interim/company_overview.csv', dtype=str, keep_default_na=False); strat = dict(zip(O_.corp_code, O_.induty_code.str[:2]))
    a, b = map(int, A.draws.split('-'))
    for k in range(a, b + 1):
        d = R / f'{A.model}_{k:02d}'
        if (d / PD.sfile('M1+il').name).exists(): continue
        feats(d, rewire(PS, k, strat)); fit(d, ['M1+il']); print(A.model, k, 'done', flush=True)
else:
    deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t in (2021, 2022, 2023) for g in [pd.read_csv(L / f'data_v3x/graph_{t}.csv', dtype=str)]}
    def per_pos(f):
        S = pd.read_csv(f, dtype={'pair_id': str, 'anchor': str, 'other': str}); r = {}
        for pid, g in S.groupby('pair_id'):
            p = g[g.label == 1].iloc[0]; n = g.score.values[g.label.values == 0]; s = p.score
            r[pid] = (p.fold, p.anchor, min(deg[p.fy][p.anchor], deg[p.fy][p.other]), ((n < s).sum() + .5 * (n == s).sum()) / len(n))
        return pd.DataFrame(r, index=['fold', 'anchor', 'md', 'auc']).T
    base = per_pos(R / 'real' / 'scores_M1.csv'); W = base[['fold', 'anchor', 'md']].copy(); W['md'] = W.md.astype(int); W['cl'] = W.fold + '-' + W.anchor
    for v in ('M1', 'M1+il', 'M1+ms', 'M1+il+ms'): W[v] = per_pos(R / 'real' / PD.sfile(v).name).auc.astype(float)
    GR = {'low': (W.md <= 1).values, 'iso': (W.md == 0).values, 'all': np.ones(len(W), bool)}
    cl = W.cl.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(20261103)
    DR = [np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]) for _ in range(1000)]
    ci = lambda d, m: [round(float(d[m].mean()), 4), round(float(np.percentile([d[x][m[x]].mean() for x in DR], 2.5)), 4), round(float(np.percentile([d[x][m[x]].mean() for x in DR], 97.5)), 4)]
    res = dict(n={g: int(m.sum()) for g, m in GR.items()}, real_gain={g: ci((W['M1+il'] - W['M1']).values, m) for g, m in GR.items()},
               marginals=dict(il_beyond_marginals={g: ci((W['M1+il+ms'] - W['M1+ms']).values, m) for g, m in GR.items()},
                              marginals_alone={g: ci((W['M1+ms'] - W['M1']).values, m) for g, m in GR.items()}), null={})
    for mdl in ('N1', 'N2'):
        ds = sorted(R.glob(f'{mdl}_*'))
        if not ds: continue
        Z = pd.DataFrame({d.name: per_pos(d / 'scores_M1+il.csv').auc.astype(float) for d in ds}).reindex(W.index)
        G0 = {g: [float((Z[c] - W['M1']).values[m].mean()) for c in Z] for g, m in GR.items()}
        ex = (W['M1+il'] - Z.mean(axis=1)).values; out = {}
        for g, m in GR.items():
            Gr = float((W['M1+il'] - W['M1']).values[m].mean())
            out[g] = dict(real=round(Gr, 4), null_mean=round(float(np.mean(G0[g])), 4), null_sd=round(float(np.std(G0[g], ddof=1)), 4), null_min=round(min(G0[g]), 4), null_max=round(max(G0[g]), 4),
                          p=round((1 + sum(x >= Gr for x in G0[g])) / (len(G0[g]) + 1), 4), excess=ci(ex, m), draws=len(G0[g]))
        res['null'][mdl] = out
    if 'N1' in res['null']:
        o = res['null']['N1']['low']; res['reading'] = '具体配对有信息' if o['p'] <= 1 / 21 + 1e-9 and o['excess'][1] > 0 else ('部分支持' if o['excess'][0] > 0 else '没有证据')
    json.dump(res, open(R / 'rewire_eval.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False); print(json.dumps(res, ensure_ascii=False, indent=1))
