"""SEAL（Zhang & Chen 2018）按论文重写，**不是官方实现**（DEV_RESULTS 第 18.3 节）。
- 子图：目标点对 (u, v) 的 1 跳封闭子图 = {u, v} ∪ N(u) ∪ N(v) 的导出子图；每个节点最多取 MAXNB = 20 个邻居（固定抽样，种子 0）；
  子图里删去目标边 (u, v)（训练正例在图中时）。
- 节点标记（h = 1 时 DRNL 的等价形式，5 类）：0 = u，1 = v，2 = 共同邻居，3 = 只与 u 相邻，4 = 只与 v 相邻（算到一端的距离时去掉另一端）。
- 网络：输入 [节点特征, 标记 one-hot] → 线性层 → 3 层 GCN（对称归一化、带自环）→ 读出 [h_u, h_v, 均值池化] → MLP。
- 训练：Adam，batch 512，最多 50 轮，按验证组内 AUC 早停（patience 5）。
子图存成拼接数组（节点、标记、局部边及偏移），每个 batch 用向量化的区间索引取出。"""
import time
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as F

MAXNB = 20; NLAB = 5

def ranges(starts, lens):
    """把若干 [start, start+len) 区间拼成一个索引数组。"""
    if len(lens) == 0: return np.zeros(0, np.int64)
    c = np.cumsum(lens); return np.repeat(starts - np.r_[0, c[:-1]], lens) + np.arange(c[-1])

class Graph:
    def __init__(s, n, edges, seed=0):
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        A = sp.csr_matrix((np.ones(2 * len(e), np.int8), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)); A.sum_duplicates(); A.data[:] = 1
        rng = np.random.default_rng(seed); s.A = A; s.nb = []
        for i in range(n):
            a = A.indices[A.indptr[i]:A.indptr[i + 1]]
            s.nb.append(a if len(a) <= MAXNB else np.sort(rng.choice(a, MAXNB, replace=False)))

def extract(g, U, V, off=0):
    """对每个点对 (U[k], V[k]) 提取封闭子图。返回 dict：nodes（全局节点号 + off）、lab、noff、e（局部边，2 × E）、eoff。"""
    nodes, labs, nl, es, el = [], [], [], [], []
    for u, v in zip(U, V):
        Nu = g.nb[u][g.nb[u] != v]; Nv = g.nb[v][g.nb[v] != u]
        oth = np.union1d(Nu, Nv); oth = oth[(oth != u) & (oth != v)]
        S = np.r_[u, v, oth]; iu = np.isin(oth, Nu); iv = np.isin(oth, Nv)
        lab = np.r_[0, 1, np.where(iu & iv, 2, np.where(iu, 3, 4))]
        sub = g.A[S][:, S].tocoo(); keep = ~(((sub.row == 0) & (sub.col == 1)) | ((sub.row == 1) & (sub.col == 0)))
        nodes.append(S + off); labs.append(lab); nl.append(len(S)); es.append(np.vstack([sub.row[keep], sub.col[keep]])); el.append(int(keep.sum()))
    return dict(nodes=np.concatenate(nodes).astype(np.int64), lab=np.concatenate(labs).astype(np.int64), noff=np.r_[0, np.cumsum(nl)],
                e=np.concatenate(es, 1).astype(np.int64), eoff=np.r_[0, np.cumsum(el)])

def concat(stores):
    out = dict(nodes=np.concatenate([s['nodes'] for s in stores]), lab=np.concatenate([s['lab'] for s in stores]), e=np.concatenate([s['e'] for s in stores], 1))
    nl = np.concatenate([np.diff(s['noff']) for s in stores]); el = np.concatenate([np.diff(s['eoff']) for s in stores])
    out['noff'] = np.r_[0, np.cumsum(nl)]; out['eoff'] = np.r_[0, np.cumsum(el)]; return out

def batch(st, idx):
    """取出第 idx 个子图，拼成一个大图：节点全局号、标记、边（已加批内偏移）、每个节点所属子图、每个子图第一个节点（u）的位置。"""
    ns, nl = st['noff'][idx], st['noff'][idx + 1] - st['noff'][idx]; es, el = st['eoff'][idx], st['eoff'][idx + 1] - st['eoff'][idx]
    pn = ranges(ns, nl); pe = ranges(es, el); first = np.r_[0, np.cumsum(nl)[:-1]]
    e = st['e'][:, pe] + np.repeat(first, el)
    return st['nodes'][pn], st['lab'][pn], e, np.repeat(np.arange(len(idx)), nl), first

class SEAL(nn.Module):
    def __init__(s, fin, hid=32, layers=3, drop=0.3):
        super().__init__(); s.inp = nn.Linear(fin + NLAB, hid); s.convs = nn.ModuleList(nn.Linear(hid, hid) for _ in range(layers))
        s.drop = nn.Dropout(drop); s.out = nn.Sequential(nn.Linear(3 * hid, hid), nn.ReLU(), nn.Linear(hid, 1))
    def forward(s, x, st, idx):
        nodes, lab, e, seg, first = batch(st, idx); n = len(nodes); dev = x.device
        nodes, lab, seg, first = (torch.as_tensor(a, device=dev) for a in (nodes, lab, seg, first)); e = torch.as_tensor(e, device=dev)
        r = torch.cat([e[0], torch.arange(n, device=dev)]); c = torch.cat([e[1], torch.arange(n, device=dev)])
        d = torch.zeros(n, device=dev).index_add_(0, r, torch.ones_like(r, dtype=torch.float32)); w = 1 / torch.sqrt(d[r] * d[c])
        A = torch.sparse_coo_tensor(torch.stack([r, c]), w, (n, n)).coalesce()
        h = torch.relu(s.inp(torch.cat([x[nodes], F.one_hot(lab, NLAB).float()], -1)))
        for conv in s.convs: h = s.drop(torch.relu(conv(torch.sparse.mm(A, h))))
        B = len(idx); cnt = torch.zeros(B, device=dev).index_add_(0, seg, torch.ones(n, device=dev))
        pool = torch.zeros(B, h.shape[1], device=dev).index_add_(0, seg, h) / cnt[:, None]
        return s.out(torch.cat([h[first], h[first + 1], pool], -1)).squeeze(-1)

def score(m, x, st, bs=4096):
    m.eval(); n = len(st['noff']) - 1; out = []
    with torch.no_grad():
        for i in range(0, n, bs): out.append(torch.sigmoid(m(x, st, np.arange(i, min(i + bs, n)))).cpu().numpy())
    return np.concatenate(out)

def train(x, st_tr, y_tr, st_va, auc_va, st_te, seed, hid, lr, max_ep=50, pat=5, bs=512):
    """训练一个 SEAL：返回（最佳验证 AUC，最佳轮的验证分数，最佳轮的测试分数；st_te 为 None 时测试分数为 None）。"""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = SEAL(x.shape[1], hid); opt = torch.optim.Adam(m.parameters(), lr=lr); y = torch.as_tensor(y_tr, dtype=torch.float32, device=x.device)
    best, bad, sv, stt = -1, 0, None, None; ntr = len(y_tr)
    for ep in range(max_ep):
        m.train(); perm = rng.permutation(ntr)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            F.binary_cross_entropy_with_logits(m(x, st_tr, idx), y[torch.as_tensor(idx, device=x.device)]).backward(); opt.step()
        v = score(m, x, st_va); a = auc_va(v)
        if a > best: best, bad, sv, stt = a, 0, v, (score(m, x, st_te) if st_te is not None else None)
        else:
            bad += 1
            if bad >= pat: break
    return best, sv, stt

# ---------------- 韩国数据（paper_dev.run_base 调用） ----------------
_CACHE = {}
def run_seal_kr(X, Ys, tr_years, va_year, seed, cfg, fast_auc):
    """韩国数据：每年用 G_t（无向）提取子图，节点特征用 M1 的节点特征（各年拼接成一个矩阵）。返回（验证分数，最佳验证 AUC）。"""
    key = (tuple(tr_years), va_year)
    if key not in _CACHE:
        t0 = time.time(); years = list(tr_years) + [va_year]; off, xs = {}, []; o = 0
        for t in years: off[t] = o; xs.append(Ys[t]['x']); o += len(Ys[t]['ids'])
        def st_of(df):
            parts = []
            for t, g in df.groupby('fy', sort=False):
                assert (np.diff(g.index.values) > 0).all()
                G = Graph(len(Ys[t]['ids']), Ys[t]['edges']); idx = Ys[t]['idx']
                parts.append((g.index.values, extract(G, [idx[a] for a in g.anchor], [idx[b] for b in g.other], off[t])))
            order = np.argsort(np.concatenate([p[0] for p in parts]), kind='stable'); st = concat([p[1] for p in parts])
            assert (order == np.arange(len(order))).all(), 'pairs must be grouped by year in index order'
            return st
        TR = X[X.fy.isin(tr_years)].sort_index(); VA = X[X.fy == va_year].sort_index()
        _CACHE[key] = (np.vstack(xs).astype(np.float32), st_of(TR), TR.label.values, st_of(VA), VA, round(time.time() - t0, 1))
        print('SEAL subgraphs', key, 'train', len(TR), 'val', len(VA), f'{_CACHE[key][-1]}s', flush=True)
    xa, st_tr, y_tr, st_va, VA, _ = _CACHE[key]
    best, sv, _ = train(torch.as_tensor(xa), st_tr, y_tr, st_va, lambda s: fast_auc(VA, s), None, seed, cfg['hid'], cfg['lr'])
    return sv, best

SEAL_GRID = [dict(hid=h, lr=lr) for h in (32, 64) for lr in (1e-3, 1e-2)]
