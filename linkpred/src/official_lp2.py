"""DEV_RESULTS 第 26.3 节：NCN（GraphPKU/NeuralCommonNeighbor）与 BUDDY（melifluos/subgraph-sketching）官方代码接入韩国数据（只用开发年三折）。
官方代码在 D:\\claude data\\official_code\\（仓库外，不修改）。须在独立虚拟环境 D:\\claude data\\venvs\\offlp 中运行（torch 2.4.1+cpu、torch_sparse），不用冻结的 enet。
用法：python src/official_lp2.py run --model ncn|buddy [--seeds 10]
      python src/official_lp2.py eval     # 写 out_dev_gpu10/official_lp2_eval.json"""
import argparse, collections, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
OC = Path(r'D:\claude data\official_code')
ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['run', 'eval']); ap.add_argument('--model', choices=['ncn', 'buddy']); ap.add_argument('--seeds', type=int, default=10)
ap.add_argument('--smoke', type=int, default=0, help='只用于冒烟测试：限制轮数、只跑 F1 一个种子')
A = ap.parse_args()
argv0 = sys.argv; sys.argv = ['paper_dev.py', 'none', '--out', 'out_dev_gpu10', '--device', 'cpu']
import paper_dev as PD
sys.argv = argv0; torch = PD.torch; O = PD.O; L = PD.L

def ei_of(Y):
    e = np.array(Y['edges'], dtype=np.int64).reshape(-1, 2)
    return torch.tensor(np.r_[e, e[:, ::-1]].T.copy(), dtype=torch.long)

def pairs_of(Ys, D):
    return {t: (np.array([Ys[t]['idx'][a] for a in g.anchor]), np.array([Ys[t]['idx'][o] for o in g.other]), g.label.values, g.index.values) for t, g in D.groupby('fy')}

def one_neg(D, rng):
    lab = D.label.values; pos = np.where(lab == 1)[0]; keep = []
    for p in pos: keep += [p, p + 1 + rng.integers(100)]
    return D.iloc[sorted(keep)]

def run_ncn(X, Ys, tr, va, seed):
    sys.path.insert(0, str(OC / 'NeuralCommonNeighbor'))
    from model import predictor_dict, GCN; from torch_sparse import SparseTensor; from functools import partial
    F = torch.nn.functional
    P = argparse.Namespace(hiddim=256, mplayers=1, xdp=0.7, tdp=0.3, gnnedp=0.0, preedp=0.4, predp=0.05, gnndp=0.05, gnnlr=0.0043, prelr=0.0024, batch_size=1152,
                           ln=True, lnnn=True, jk=True, use_xlin=True, tailact=True, twolayerlin=False, beta=1.0, nnlayers=3, cndeg=-1, epochs=100, res=False)
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    years = list(tr) + [va]; n = {t: len(Ys[t]['ids']) for t in years}
    xs = {t: torch.tensor(Ys[t]['x'], dtype=torch.float32) for t in years}
    adj = {t: SparseTensor.from_edge_index(ei_of(Ys[t]), sparse_sizes=(n[t], n[t])).to_symmetric().coalesce() for t in years}
    model = GCN(xs[va].shape[1], P.hiddim, P.hiddim, P.mplayers, P.gnndp, P.ln, P.res, -1, 'puregcn', P.jk, P.gnnedp, xdropout=P.xdp, taildropout=P.tdp)
    predictor = partial(predictor_dict['cn1'], cndeg=P.cndeg, use_xlin=P.use_xlin, tailact=P.tailact, twolayerlin=P.twolayerlin, beta=P.beta)(P.hiddim, P.hiddim, 1, P.nnlayers, P.predp, P.preedp, P.lnnn)
    opt = torch.optim.Adam([{'params': model.parameters(), 'lr': P.gnnlr}, {'params': predictor.parameters(), 'lr': P.prelr}])
    TRall = X[X.fy.isin(tr)]; VA = X[X.fy == va]; u_va, v_va, _, ix_va = pairs_of(Ys, VA)[va]
    best, state = -1, None
    for ep in range(A.smoke or P.epochs):
        model.train(); predictor.train(); TR = one_neg(TRall, rng)   # 官方每轮重新做负采样
        for t, (u, v, l, _) in pairs_of(Ys, TR).items():
            pos = torch.tensor(np.stack([u[l == 1], v[l == 1]])); neg = torch.tensor(np.stack([u[l == 0], v[l == 0]])); perm_all = torch.randperm(pos.shape[1])
            for s in range(0, pos.shape[1], P.batch_size):
                perm = perm_all[s:s + P.batch_size]; opt.zero_grad(); h = model(xs[t], adj[t])
                loss = -F.logsigmoid(predictor.multidomainforward(h, adj[t], pos[:, perm], cndropprobs=[])).mean() - F.logsigmoid(-predictor.multidomainforward(h, adj[t], neg[:, perm], cndropprobs=[])).mean()
                loss.backward(); opt.step()
        model.eval(); predictor.eval()
        with torch.no_grad():
            h = model(xs[va], adj[va]); s_ = torch.sigmoid(predictor(h, adj[va], torch.tensor(np.stack([u_va, v_va])))).squeeze(-1).numpy()
        a = PD.fast_auc(VA, s_)
        if a > best: best, state = a, s_
    return state

def run_buddy(X, Ys, tr, va, seed):
    sys.path.insert(0, str(OC / 'subgraph-sketching'))
    from src.models.elph import BUDDY; from src.hashing import ElphHashes; from src.datasets.elph import HashDataset
    from torch_geometric.data import Data
    P = argparse.Namespace(use_feature=True, label_dropout=0.5, feature_dropout=0.5, propagate_embeddings=False, add_normed_features=None, use_RA=False, max_hash_hops=2, sign_k=0,
                           sign_dropout=0.5, hidden_channels=1024, hll_p=8, minhash_num_perm=128, floor_sf=0, use_zero_one=0, lr=1e-4, batch_size=1024, epochs=100)
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    years = list(tr) + [va]; feat = {}
    TR = one_neg(X[X.fy.isin(tr)], rng); VA = X[X.fy == va]   # 官方：训练负例抽一次
    PR = {**pairs_of(Ys, TR), va: pairs_of(Ys, VA)[va]}
    for t in years:
        n = len(Ys[t]['ids']); ei = ei_of(Ys[t]); x = torch.tensor(Ys[t]['x'], dtype=torch.float32)
        xs = HashDataset._generate_sign_features(None, Data(x=x), ei, torch.ones(ei.shape[1]), 0)
        H = ElphHashes(P); table, cards = H.build_hash_tables(n, ei)
        u, v, l, ix = PR[t]; links = torch.tensor(np.stack([u, v], 1))
        sf = H.get_subgraph_features(links, table, cards).float()
        deg = torch.zeros(n).index_add_(0, ei[0], torch.ones(ei.shape[1]))
        feat[t] = (sf, xs[links], deg[links[:, 0]], deg[links[:, 1]], torch.tensor(l, dtype=torch.float32))
    model = BUDDY(P, num_features=feat[va][1].shape[-1]); opt = torch.optim.Adam(model.parameters(), lr=P.lr)
    trs = [feat[t] for t in tr]; sf_t = torch.cat([f[0] for f in trs]); xf_t = torch.cat([f[1] for f in trs]); d1 = torch.cat([f[2] for f in trs]); d2 = torch.cat([f[3] for f in trs]); y = torch.cat([f[4] for f in trs])
    best, state = -1, None
    for ep in range(A.smoke or P.epochs):
        model.train(); perm_all = torch.randperm(len(y))
        for s in range(0, len(y), P.batch_size):
            i = perm_all[s:s + P.batch_size]
            if len(i) < 2: continue
            opt.zero_grad(); logit = model(sf_t[i], xf_t[i], d1[i], d2[i], None, None).squeeze(-1)
            torch.nn.functional.binary_cross_entropy_with_logits(logit, y[i]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            sv, xv, a1, a2, _ = feat[va]; s_ = torch.sigmoid(model(sv, xv, a1, a2, None, None).squeeze(-1)).numpy()
        a = PD.fast_auc(VA, s_)
        if a > best: best, state = a, s_
    return state

if A.cmd == 'run':
    X, N, G = PD.load(); rows = []; fn = run_ncn if A.model == 'ncn' else run_buddy
    seeds = [1] if A.smoke else list(range(1, A.seeds + 1)); folds = {'F1': PD.FOLDS['F1']} if A.smoke else PD.FOLDS
    for f, (tr, va) in folds.items():
        Ys = PD.node_tables(N, G, list(tr) + [va], tr, False); VA = X[X.fy == va]; t0 = time.time()
        per = {f'score_s{sd}': fn(X, Ys, tr, va, sd) for sd in seeds}; s = np.mean(list(per.values()), 0)
        rows.append(VA[['pair_id', 'fy', 'anchor', 'other', 'label']].assign(fold=f, score=s, **per))
        print(A.model, f, round(PD.fast_auc(VA, s), 4), f'{(time.time() - t0) / len(seeds):.0f}s/seed', flush=True)
    out = O / f'scores_{A.model}-official.csv' if not A.smoke else O / f'_smoke_{A.model}.csv'
    pd.concat(rows).to_csv(out, index=False); print('written', out.name)
else:
    deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t in (2021, 2022, 2023) for g in [pd.read_csv(L / f'data_v3x/graph_{t}.csv', dtype=str)]}
    def per_pos(f):
        S = pd.read_csv(f, dtype={'pair_id': str, 'anchor': str, 'other': str}); sc = [c for c in S.columns if c.startswith('score')]; r = {}; meta = {}
        for pid, g in S.groupby('pair_id'):
            p = g[g.label == 1].iloc[0]; lab = g.label.values; meta[pid] = (p.fold, p.anchor, deg[p.fy][p.anchor], deg[p.fy][p.other])
            for c in sc:
                v = g[c].values; s = v[lab == 1][0]; n = v[lab == 0]; r.setdefault(c, {})[pid] = ((n < s).sum() + .5 * (n == s).sum()) / len(n)
        return pd.DataFrame(r), pd.DataFrame(meta, index=['fold', 'anchor', 'da', 'do']).T
    MS = ['ncn-official', 'buddy-official', 'dfgt', 'dfgt-nograph', 'M1', 'sage']
    AU = {}; M = None
    for v in MS:
        AU[v], meta = per_pos(O / f'scores_{v}.csv'); M = meta if M is None else M
    M['da'] = M.da.astype(int); M['do'] = M['do'].astype(int); M['md'] = M[['da', 'do']].min(axis=1); M['cl'] = M.fold + '-' + M.anchor
    GR = {'low': (M.md <= 1).values, 'iso': (M.md == 0).values, 'high': (M.md >= 2).values, 'all': np.ones(len(M), bool),
          'iso_anchor': (M.da == 0).values, 'iso_only_other': ((M.da > 0) & (M['do'] == 0)).values}
    cl = M.cl.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(20261103); DR = [np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]) for _ in range(1000)]
    ci = lambda d, m: [round(float(d[m].mean()), 4), round(float(np.percentile([d[i][m[i]].mean() for i in DR if m[i].any()], 2.5)), 4), round(float(np.percentile([d[i][m[i]].mean() for i in DR if m[i].any()], 97.5)), 4)]
    res = dict(n={g: int(m.sum()) for g, m in GR.items()}, auc={}, single={}, diff={})
    for v in MS:
        s = AU[v].score.reindex(M.index).values; res['auc'][v] = {g: round(float(s[m].mean()), 4) for g, m in GR.items()}; res['auc'][v]['gap'] = round(res['auc'][v]['high'] - res['auc'][v]['low'], 4)
        sc = [c for c in AU[v].columns if c != 'score']
        if sc:
            per = pd.DataFrame([{g: float(AU[v][c].reindex(M.index).values[m].mean()) for g, m in GR.items()} for c in sc])
            res['single'][v] = {g: [round(float(per[g].mean()), 4), round(float(per[g].std(ddof=1)), 4)] for g in per}
    for a in ('ncn-official', 'buddy-official'):
        for b in ('dfgt', 'M1', 'dfgt-nograph', 'sage'):
            d = (AU[a].score - AU[b].score).reindex(M.index).values; res['diff'][f'{a} − {b}'] = {g: ci(d, m) for g, m in GR.items()}
    json.dump(res, open(O / 'official_lp2_eval.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False); print(json.dumps(res, ensure_ascii=False, indent=1))
