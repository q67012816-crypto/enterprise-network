"""DEV_RESULTS 第 20.5 节：官方代码（Tail-GNN、SEAL_OGB）接入韩国数据，与重写版本对比（只用开发年的三折）。
官方代码在 D:\\claude data\\official_code\\（仓库外，不修改）。本文件只做数据接入：导入官方模型、损失和子图函数，
接入我们的年度图 G_t、M1 节点特征和点对；按验证年组内 AUC 早停；分数写 out_dev_gpu10/scores_<name>-official.csv。
用法：python src/official_kr.py run --model tailgnn|seal --device cuda
      python src/official_kr.py eval"""
import argparse, copy, os, sys, time
SMOKE = int(os.environ.get('OFFICIAL_SMOKE_EPOCHS', '0'))   # 只用于冒烟测试：限制轮数、只跑 F1 一个种子；正式运行不设
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
OC = Path(r'D:\claude data\official_code')
ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['run', 'eval']); ap.add_argument('--model', choices=['tailgnn', 'seal'])
ap.add_argument('--device', default='cuda'); A = ap.parse_args()
SEEDS = {'tailgnn': list(range(1, 11)), 'seal': [1, 2, 3]}
argv0 = sys.argv; sys.argv = ['paper_dev.py', 'none', '--out', 'out_dev_gpu10', '--device', 'cpu']
import paper_dev as PD
sys.argv = argv0; torch = PD.torch; O = PD.O
dev = A.device if torch is not None and torch.cuda.is_available() else 'cpu'

def year_adj(Y):
    import scipy.sparse as ssp
    n = len(Y['ids']); e = np.array(Y['edges'], dtype=np.int64).reshape(-1, 2)
    M = ssp.csr_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)); M.data[:] = 1
    return M

def pairs_of(Ys, D):
    return {t: (np.array([Ys[t]['idx'][a] for a in g.anchor]), np.array([Ys[t]['idx'][o] for o in g.other]), g.label.values, g.index.values) for t, g in D.groupby('fy')}

def one_neg(D, rng):
    """每个正例配 1 个负例：从该正例紧跟的 100 个候选负例中随机取一个（点对按"正例 + 100 个负例"成块排列）。"""
    lab = D.label.values; pos = np.where(lab == 1)[0]; keep = []
    for p in pos: keep += [p, p + 1 + rng.integers(100)]
    return D.iloc[sorted(keep)]

# ---------------- Tail-GNN（官方模型、判别器、损失、超参数） ----------------
def run_tailgnn(X, Ys, tr, va, seed):
    sys.path.insert(0, str(OC / 'Tail-GNN')); sys.path.insert(0, str(OC / 'Tail-GNN' / 'link_prediction'))
    from models import TailGNN; from layers import Discriminator; import link_data_process as LDP
    nn, F = torch.nn, torch.nn.functional
    P = argparse.Namespace(hidden=32, dropout=0.5, arch=1, g_sigma=1, ablation=0, eta=1, mu=0.001, lamda=1e-4, lr=0.01, k=5, epochs=1000, patience=300)
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed); crit = nn.BCELoss(); nclass = 16
    G = {}
    for t in list(tr) + [va]:
        Mt = year_adj(Ys[t]).tolil(); head, _ = LDP.split_nodes(Mt, k=P.k); head = np.asarray(head).reshape(-1)
        tail = LDP.link_dropout(Mt, head, k=P.k)
        G[t] = (torch.tensor(Ys[t]['x'], device=dev), torch.FloatTensor(Mt.todense()).to(dev), torch.FloatTensor(tail.todense()).to(dev))
    TR = one_neg(X[X.fy.isin(tr)], rng); VA = X[X.fy == va]
    trp = {t: (torch.tensor(u[l == 1], device=dev), torch.tensor(v[l == 1], device=dev), torch.tensor(v[l == 0], device=dev)) for t, (u, v, l, _) in pairs_of(Ys, TR).items()}
    vap = pairs_of(Ys, VA)[va]
    m = TailGNN(nfeat=G[va][0].shape[1], nclass=nclass, params=P, device=dev, ver=2).to(dev)
    disc = Discriminator(nclass).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=P.lr, weight_decay=P.lamda); optD = torch.optim.Adam(disc.parameters(), lr=P.lr, weight_decay=P.lamda)
    norm_out = lambda out, idx: sum(torch.mean(torch.norm(z[idx], dim=1)) for z in out)
    best, bad, state = -1, 0, None
    for ep in range(SMOKE or P.epochs):
        for _ in range(2):   # 官方：每轮判别器 2 步
            disc.train(); optD.zero_grad(); L = 0.
            for t, (b, p, n_) in trp.items():
                x, adj, tadj = G[t]; eh, _, _ = m(x, adj, True); et, _, _ = m(x, tadj, False)
                L = L + (crit(disc(eh)[b], torch.ones(len(b), 1, device=dev)) + crit(disc(et)[b], torch.zeros(len(b), 1, device=dev))) / 2
            (L / len(trp)).backward(); optD.step()
        m.train(); opt.zero_grad(); L = 0.
        for t, (b, p, n_) in trp.items():
            x, adj, tadj = G[t]; eh, _, sh = m(x, adj, True); et, _, st = m(x, tadj, False)
            lh = -F.logsigmoid((eh[b] * eh[p]).sum(1) - (eh[b] * eh[n_]).sum(1)).mean(); lt = -F.logsigmoid((et[b] * et[p]).sum(1) - (et[b] * et[n_]).sum(1)).mean()
            Ld = crit(disc(et)[b], torch.zeros(len(b), 1, device=dev))
            L = L + (lh + lt) / 2 - P.eta * Ld + P.mu * norm_out(sh, b)
        (L / len(trp)).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            x, adj, _ = G[va]; e, _, _ = m(x, adj, False); u, v, _, ix = vap
            s = torch.sigmoid((e[torch.tensor(u, device=dev)] * e[torch.tensor(v, device=dev)]).sum(1)).cpu().numpy()
        a = PD.fast_auc(VA, s)
        if a >= best: best, bad, state = a, 0, s   # 官方用 ">="
        else:
            bad += 1
            if bad >= P.patience: break
    return state

# ---------------- SEAL（官方子图提取、DRNL、DGCNN、默认超参数） ----------------
def run_seal(X, Ys, tr, va, seed, cache={}):
    sys.path.insert(0, str(OC / '_shims')); sys.path.insert(0, str(OC / 'SEAL_OGB'))
    import utils as SU; from models import DGCNN; from torch_geometric.loader import DataLoader
    import random; random.seed(seed); torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    def build(D):
        out = []
        for t, (u, v, l, ix) in pairs_of(Ys, D).items():
            Mt = year_adj(Ys[t]); xt = torch.tensor(Ys[t]['x'])
            for a_, b_, y_ in zip(u, v, l): out.append(SU.construct_pyg_graph(*SU.k_hop_subgraph(int(a_), int(b_), 1, Mt, 1.0, None, node_features=xt, y=int(y_)), 'drnl'))
        return out
    TR = one_neg(X[X.fy.isin(tr)], rng); VA = X[X.fy == va]
    key = (tuple(tr), va)
    if key not in cache: t0 = time.time(); cache[key] = build(VA); print('SEAL val subgraphs', key, len(cache[key]), f'{time.time() - t0:.0f}s', flush=True)
    tr_ds = build(TR); va_ds = cache[key]
    class DS(list): num_features = tr_ds[0].x.shape[1]
    m = DGCNN(32, 3, 1000, 0.6, DS(tr_ds), False, use_feature=True).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    tl = DataLoader(tr_ds, batch_size=32, shuffle=True); vl = DataLoader(va_ds, batch_size=256, shuffle=False)
    best, state = -1, None
    for ep in range(SMOKE or 50):
        m.train()
        for d in tl:
            d = d.to(dev); opt.zero_grad()
            loss = torch.nn.BCEWithLogitsLoss()(m(d.z, d.edge_index, d.batch, d.x, None, None).view(-1), d.y.to(torch.float)); loss.backward(); opt.step()
        m.eval(); s = []
        with torch.no_grad():
            for d in vl: d = d.to(dev); s.append(torch.sigmoid(m(d.z, d.edge_index, d.batch, d.x, None, None).view(-1)).cpu().numpy())
        s = np.concatenate(s); a = PD.fast_auc(VA, s)
        if a > best: best, state = a, s
    return state

if A.cmd == 'run':
    X, N, G = PD.load(); rows = []; fn = {'tailgnn': run_tailgnn, 'seal': run_seal}[A.model]
    for f, (tr, va) in (list(PD.FOLDS.items())[:1] if SMOKE else PD.FOLDS.items()):
        Ys = PD.node_tables(N, G, list(tr) + [va], tr, False); VA = X[X.fy == va]; t0 = time.time()
        per = {f'score_s{sd}': fn(X, Ys, tr, va, sd) for sd in (SEEDS[A.model][:1] if SMOKE else SEEDS[A.model])}; s = np.mean(list(per.values()), 0)
        rows.append(VA[['pair_id', 'fy', 'anchor', 'other', 'label']].assign(fold=f, score=s, **per))
        print(A.model, 'official', f, round(PD.fast_auc(VA, s), 4), f'{(time.time() - t0) / len(SEEDS[A.model]):.0f}s/seed', flush=True)
    if not SMOKE: pd.concat(rows).to_csv(PD.sfile(f'{A.model}-official'), index=False)
else:
    import collections, json
    X, N, G = PD.load(); deg = {t: collections.Counter(list(G[t].src) + list(G[t].dst)) for t in G}
    MS = ['dfgt', 'tailgnn', 'tailgnn-official', 'seal', 'seal-official']; W = None
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
    for a, b in [('tailgnn-official', 'tailgnn'), ('seal-official', 'seal'), ('tailgnn-official', 'dfgt'), ('seal-official', 'dfgt'), ('tailgnn', 'dfgt'), ('seal', 'dfgt')]:
        d = (W[a] - W[b]).values; res['diff'][f'{a} − {b}'] = {g: [round(float(d[m].mean()), 4)] + [round(float(q), 4) for q in np.percentile([d[dr][m[dr]].mean() for dr in draws], [2.5, 97.5])] for g, m in grp.items()}
    json.dump(res, open(O / 'official_eval.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(pd.DataFrame(res['auc']).T.to_string())
    for k, v in res['diff'].items(): print(f'{k:28s}', '  '.join(f'{g}: {x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]' for g, x in v.items()))
