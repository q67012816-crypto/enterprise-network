"""DEV_RESULTS 第 23 节：选轮次的乐观偏差（只用开发年；不改冻结的 paper_dev.py）。
  诚实臂：训练 tr，在 sel 年选轮次（早停），用选中的参数给 te 年只打分一次（paper_dev.run_gnn 的 te_year 选项）。
  乐观臂：同样的训练年和种子，直接在 te 年选轮次并报告 te 年分数（现行做法）。
用法：
  python src/dev_holdout.py run --device cuda     # 写 out_dev_holdout/{honest,oracle}/scores_<model>.csv（fold = H2 / H3，含每个种子的分数）
  python src/dev_holdout.py eval                  # 写 out_dev_holdout/holdout_eval.json"""
import argparse, collections, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['run', 'eval']); ap.add_argument('--device', default='cuda'); ap.add_argument('--models', default='')
A = ap.parse_args()
HOLD = {'H2': ((2019, 2020), 2021, 2022), 'H3': ((2019, 2020, 2021), 2022, 2023)}
GNN = ['dfgt', 'dfgt-nograph', 'sage', 'dfgt+port+fine', 'dfgt+port+fine+il']; LR = ['M1', 'M1+il']; SEEDS = list(range(1, 11))
argv0 = sys.argv; sys.argv = ['paper_dev.py', 'none', '--out', 'out_dev_holdout', '--seeds', '10', '--device', A.device if A.cmd == 'run' else 'cpu']
import paper_dev as PD
sys.argv = argv0; O = PD.O; L = PD.L
for arm in ('honest', 'oracle'): (O / arm).mkdir(exist_ok=True)
cols = ['pair_id', 'fy', 'anchor', 'other', 'label']

if A.cmd == 'run':
    X, N, G = PD.load(); models = A.models.split(',') if A.models else LR + GNN
    for v in models:
        model, opt = PD.VARIANTS[v]; out = {'honest': [], 'oracle': []}; t0 = time.time()
        for f, (tr, sel, te) in HOLD.items():
            Ys = PD.node_tables(N, G, list(tr) + [sel, te], tr, opt.get('target', False)); TE = X[X.fy == te]
            if model == 'lr':
                _, st = PD.run_lr(X, Ys, tr, sel, opt, te_year=te)
                for arm in out: out[arm].append(TE[cols].assign(fold=f, score=st))
                continue
            hon = {f'score_s{sd}': PD.run_gnn(X, Ys, tr, sel, model, opt, sd, te_year=te)[1] for sd in SEEDS}
            ora = {f'score_s{sd}': PD.run_gnn(X, Ys, tr, te, model, opt, sd) for sd in SEEDS}
            for arm, per in (('honest', hon), ('oracle', ora)):
                out[arm].append(TE[cols].assign(fold=f, score=np.mean(list(per.values()), 0), **per))
            print(v, f, 'honest', round(PD.fast_auc(TE, np.mean(list(hon.values()), 0)), 4), 'oracle', round(PD.fast_auc(TE, np.mean(list(ora.values()), 0)), 4), flush=True)
        for arm, rows in out.items(): pd.concat(rows).to_csv(O / arm / PD.sfile(v).name, index=False)
        print(v, 'done', round(time.time() - t0), 's', flush=True)
else:
    G = {t: pd.read_csv(L / f'data_v3x/graph_{t}.csv', dtype=str) for t in (2022, 2023)}; deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t, g in G.items()}
    def per_pos(f):
        S = pd.read_csv(f, dtype={'pair_id': str, 'anchor': str, 'other': str}); sc = [c for c in S.columns if c.startswith('score')]; out = {}; meta = {}
        for pid, g in S.groupby('pair_id'):
            lab = g.label.values; p = g[g.label == 1].iloc[0]; meta[pid] = (p.fold, p.anchor, min(deg[p.fy][p.anchor], deg[p.fy][p.other]))
            for c in sc:
                v = g[c].values; s = v[lab == 1][0]; n = v[lab == 0]; out.setdefault(c, {})[pid] = ((n < s).sum() + .5 * (n == s).sum()) / len(n)
        return pd.DataFrame(out), pd.DataFrame(meta, index=['fold', 'anchor', 'mindeg']).T
    AUC = {}; M = None
    for arm in ('honest', 'oracle'):
        for v in LR + GNN:
            AUC[(arm, v)], meta = per_pos(O / arm / PD.sfile(v).name); M = meta if M is None else M
    M['mindeg'] = M.mindeg.astype(int); M['cluster'] = M.fold + '-' + M.anchor
    GR = {'all': lambda U: np.ones(len(U), bool), 'low': lambda U: (U.mindeg <= 1).values, 'high': lambda U: (U.mindeg >= 2).values, 'iso': lambda U: (U.mindeg == 0).values}
    SUB = {'H3': (M.fold == 'H3').values, 'H2': (M.fold == 'H2').values, 'pooled': np.ones(len(M), bool)}
    def draws(U):
        cl = U.cluster.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(20261103)
        return [np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]) for _ in range(1000)]
    def ci(d, m, dr):
        v = np.array([d[x][m[x]].mean() for x in dr]); return [round(float(d[m].mean()), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    def gauc(s, m): r = {g: float(s[m[g]].mean()) for g in m}; r['gap'] = r['high'] - r['low']; return {k: round(x, 4) for k, x in r.items()}
    res = {}
    for sname, sm in SUB.items():
        U = M[sm]; idx = U.index; dr = draws(U); mk = {g: GR[g](U) for g in GR}; R = res[sname] = dict(n={g: int(m.sum()) for g, m in mk.items()}, auc={}, optimism={}, comparisons={})
        for v in LR + GNN:
            for arm in ('honest', 'oracle'):
                P = AUC[(arm, v)].reindex(idx); e = dict(ensemble=gauc(P.score.values, mk))
                sc = [c for c in P.columns if c != 'score']
                if sc: per = pd.DataFrame([gauc(P[c].values, mk) for c in sc]); e.update(single_mean=per.mean().round(4).to_dict(), single_sd=per.std(ddof=1).round(4).to_dict())
                R['auc'].setdefault(v, {})[arm] = e
            if v in GNN:
                d = (AUC[('oracle', v)].score - AUC[('honest', v)].score).reindex(idx).values; R['optimism'][v] = {g: ci(d, mk[g], dr) for g in ('all', 'low', 'high', 'iso')}
        for a, b in (('dfgt', 'M1'), ('dfgt', 'dfgt-nograph'), ('dfgt+port+fine+il', 'dfgt+port+fine'), ('sage', 'M1'), ('dfgt-nograph', 'M1')):
            for arm in ('honest', 'oracle'):
                d = (AUC[(arm, a)].score - AUC[(arm, b)].score).reindex(idx).values; R['comparisons'].setdefault(f'{a} − {b}', {})[arm] = {g: ci(d, mk[g], dr) for g in ('all', 'low', 'iso')}
    rd = {}
    for k in ('dfgt − M1', 'dfgt+port+fine+il − dfgt+port+fine'):
        x = res['pooled']['comparisons'][k]['honest']['low']; rd[k] = dict(honest_low=x, verdict='结论不变' if x[0] > 0 and x[1] > 0 else ('减弱' if x[0] > 0 else '反转'))
    res['reading'] = rd
    # 参照：现有 F2、F3（out_dev_gpu10，训练年多一年、在报告年上选轮次）
    ref = {}
    for v in LR + GNN:
        P, meta = None, None
        S = pd.read_csv(L / 'out_dev_gpu10' / PD.sfile(v).name, dtype={'pair_id': str, 'anchor': str, 'other': str}); S = S[S.fy.isin([2022, 2023])]
        tmp = O / '_ref.csv'; S.to_csv(tmp, index=False); P, meta = per_pos(tmp); tmp.unlink()
        meta['mindeg'] = meta.mindeg.astype(int); mk = {g: GR[g](meta) for g in GR}; ref[v] = gauc(P.score.reindex(meta.index).values, mk)
    res['reference_F2F3_pooled'] = ref
    json.dump(res, open(O / 'holdout_eval.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(json.dumps(res['reading'], ensure_ascii=False))
    for s in SUB:
        print('==', s, res[s]['n'])
        for v, x in res[s]['auc'].items(): print(f"  {v:20s} honest {x['honest']['ensemble']}  oracle {x['oracle']['ensemble']}")
        for v, x in res[s]['optimism'].items(): print(f"  optimism {v:20s}", x)
        for k, x in res[s]['comparisons'].items(): print(f"  {k:36s} honest {x['honest']}  oracle {x['oracle']}")
    print('ref F2/F3 pooled', json.dumps(ref))
