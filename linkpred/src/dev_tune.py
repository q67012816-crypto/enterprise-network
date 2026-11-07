"""DEV_RESULTS 第 20.3 节：DFGT 和 GraphSAGE 用与新基线相同的小网格调参（开发阶段敏感性分析；A8 冻结的超参数不变）。
不修改冻结的 paper_dev.py：在本进程内把模型的 hidden 和 Adam 的 lr 换成网格中的值，其余训练流程完全沿用 paper_dev.run_gnn。
用法：
  python src/dev_tune.py tune --model dfgt --device cuda     # F1、F2 × 4 个配置 × 3 个种子，按全部组验证 AUC 两折均值选；写 out_dev_gpu10/tune_{model}.csv、tune_config_{model}.json
  python src/dev_tune.py run --model dfgt --device cuda      # 选出的配置，三折 × 10 个种子，写 out_dev_gpu10/scores_{model}-tuned.csv
  python src/dev_tune.py eval                                # 调参前后：各基线 − DFGT、dfgt-tuned − dfgt、sage-tuned − sage（低连接、孤立、全部；B = 1,000）"""
import argparse, functools, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['tune', 'run', 'eval']); ap.add_argument('--model', choices=['dfgt', 'sage'])
ap.add_argument('--device', default='cuda'); ap.add_argument('--seeds', type=int, default=10); A = ap.parse_args()
GRID = [dict(hid=h, lr=lr) for h in (64, 128) for lr in (1e-3, 1e-2)]   # 与 baselines.BASE_GRID 相同
argv0 = sys.argv; sys.argv = ['paper_dev.py', 'none', '--out', 'out_dev_gpu10', '--seeds', str(A.seeds if A.cmd == 'run' else 3), '--device', A.device if A.cmd != 'eval' else 'cpu']
import paper_dev as PD
sys.argv = argv0; O = PD.O; torch = PD.torch

def patch(cfg):
    """在本进程内把 SAGE / DFGTDev 的 hidden 和 Adam 的 lr 换成 cfg 中的值。"""
    from paper_models import SAGE; from dev_models import DFGTDev
    PD.SAGE = lambda fin, hid: SAGE(fin, cfg['hid']); PD.SAGE.prepare = SAGE.prepare
    PD.DFGTDev = functools.partial(DFGTDev, hid=cfg['hid'])
    adam = torch.optim.Adam.__init__
    if not hasattr(torch.optim.Adam, '_orig_init'): torch.optim.Adam._orig_init = adam
    def init(self, params, lr=1e-3, **kw): torch.optim.Adam._orig_init(self, params, lr=cfg['lr'], **kw)
    torch.optim.Adam.__init__ = init

if A.cmd == 'tune':
    X, N, G = PD.load(); rows = []; model, opt = PD.VARIANTS[A.model]
    for f in ('F1', 'F2'):
        tr, va = PD.FOLDS[f]; Ys = PD.node_tables(N, G, list(tr) + [va], tr, False); VA = X[X.fy == va]
        for k, cfg in enumerate(GRID):
            patch(cfg); t0 = time.time(); s = np.mean([PD.run_gnn(X, Ys, tr, va, model, opt, sd) for sd in (1, 2, 3)], 0)
            rows.append(dict(model=A.model, fold=f, cfg=k, **cfg, val_auc=round(PD.fast_auc(VA, s), 4), sec_per_seed=round((time.time() - t0) / 3, 1))); print(rows[-1], flush=True)
    T = pd.DataFrame(rows); T.to_csv(O / f'tune_{A.model}.csv', index=False); M = T.groupby('cfg').val_auc.mean(); k = int(M.idxmax())
    json.dump(dict(cfg=GRID[k], cfg_index=k, mean_val_auc_F1F2=round(float(M.max()), 4)), open(O / f'tune_config_{A.model}.json', 'w'), indent=1); print('selected', A.model, GRID[k])
elif A.cmd == 'run':
    cfg = json.load(open(O / f'tune_config_{A.model}.json'))['cfg']; patch(cfg)
    PD.VARIANTS[f'{A.model}-tuned'] = PD.VARIANTS[A.model]; PD.run(f'{A.model}-tuned')
else:
    X, N, G = PD.load(); import collections
    deg = {t: collections.Counter(list(G[t].src) + list(G[t].dst)) for t in G}
    MS = ['dfgt', 'dfgt-tuned', 'sage', 'sage-tuned', 'dfgt-nograph', 'M1', 'gae', 'coldbrew', 'coldbrew2', 'degfair', 'tailgnn', 'seal']
    W = None
    for v in MS:
        S = pd.read_csv(PD.sfile(v), dtype={'anchor': str, 'other': str, 'pair_id': str}); S['pos_group'] = ''
        w = PD.within_metrics(S, 'score')[['AUC']].rename(columns={'AUC': v})
        if W is None:
            pos = S[S.label == 1].set_index('pair_id'); W = w.join(pos[['anchor', 'fold', 'fy', 'other']])
            W['mindeg'] = [min(deg[t][a], deg[t][o]) for t, a, o in zip(W.fy, W.anchor, W.other)]; W['cluster'] = W.fold + '-' + W.anchor
        else: W = W.join(w)
    W = W.reset_index(); cl = W.cluster.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(20261103)
    draws = [np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]) for _ in range(1000)]
    grp = {'all': np.ones(len(W), bool), 'low': (W.mindeg <= 1).values, 'iso': (W.mindeg == 0).values}
    res = dict(auc={v: {g: round(float(W.loc[m, v].mean()), 4) for g, m in grp.items()} for v in MS}, diff={})
    pairs = [(b, 'dfgt') for b in MS[6:]] + [(b, 'dfgt-tuned') for b in MS[6:]] + [('dfgt-tuned', 'dfgt'), ('sage-tuned', 'sage'), ('dfgt-tuned', 'sage-tuned'), ('dfgt', 'sage')]
    for a, b in pairs:
        d = (W[a] - W[b]).values; R = {}
        for g, m in grp.items():
            v = [d[dr][m[dr]].mean() for dr in draws]; R[g] = [round(float(d[m].mean()), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
        res['diff'][f'{a} − {b}'] = R
    for c in ('tune_config_dfgt.json', 'tune_config_sage.json'): res[c] = json.load(open(O / c))
    json.dump(res, open(O / 'tune_eval.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(pd.DataFrame(res['auc']).T.to_string())
    for k, v in res['diff'].items(): print(f'{k:24s}', '  '.join(f'{g}: {x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]' for g, x in v.items()))
