"""度偏差论文研究（protocol.md A7）的两个图模型：GraphSAGE 和 DegFairGT 式链路预测器。纯 PyTorch，CPU 可运行。
DegFairGT 式模型按论文（arXiv 2504.15075）式 1–16 重写，不是官方代码（官方依赖 DGL）；与官方实现的差异未验证。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans

def sparse_mean_adj(n, edges):
    """无向边 → 行归一化稀疏邻接矩阵（邻居均值）。孤立节点那一行全为 0。"""
    if not edges: return torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), (n, n)).coalesce()
    e = np.array(edges); r = np.concatenate([e[:, 0], e[:, 1]]); c = np.concatenate([e[:, 1], e[:, 0]])
    deg = np.bincount(r, minlength=n).astype(np.float32)
    return torch.sparse_coo_tensor(torch.tensor(np.vstack([r, c])), torch.tensor(1 / deg[r]), (n, n)).coalesce()

class PairDecoder(nn.Module):
    """对称的 MLP 解码器：输入两家企业的表示 h_u、h_v，输出成边的 logit。用 [h_u⊙h_v, |h_u−h_v|] 保证 u、v 对称。"""
    def __init__(s, d):
        super().__init__(); s.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
    def forward(s, h, u, v):
        return s.mlp(torch.cat([h[u] * h[v], (h[u] - h[v]).abs()], -1)).squeeze(-1)

# ======================= GraphSAGE =======================
class SAGE(nn.Module):
    """GraphSAGE（均值聚合）：h' = W1·h + W2·mean(邻居 h)。输入节点特征和行归一化邻接；输出节点表示。"""
    def __init__(s, fin, hid, layers=2, drop=0.3):
        super().__init__()
        dims = [fin] + [hid] * layers
        s.self_l = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))
        s.nb_l = nn.ModuleList(nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(layers))
        s.drop = nn.Dropout(drop); s.dec = PairDecoder(hid)
    def prepare(s, Y):
        """为某一年准备只依赖图结构的输入（稀疏均值邻接）。"""
        return dict(A=sparse_mean_adj(len(Y['ids']), Y['edges']))
    def embed(s, x, P):
        h = x
        for k, (a, b) in enumerate(zip(s.self_l, s.nb_l)):
            h = a(h) + b(torch.sparse.mm(P['A'], h))
            if k < len(s.self_l) - 1: h = s.drop(F.relu(h))
        return h, torch.zeros(())          # 第二个返回值：自监督损失（SAGE 没有）

# ======================= DegFairGT 式模型 =======================
def dfgt_prepare(Y, km, xi=0.8, zeta=0.2, k_ctx=2, k_sim=3, p=3):
    """预计算一年的结构量（只依赖 G_t 和特征，不依赖标签）。
    输入：该年节点表 Y（ids、edges、deg、x）、已拟合的 k-means（社区，M=5）；
    输出：候选点对、先验概率（式 1、3：Ã=ξA+ζD，D_ij=1/√(d_i d_j) 只对同社区且 k_ctx 跳内的节点）、
          k 跳 Jaccard 相似度 s_ij（式 9–10）、D_ij、原图标记、p 步转移矩阵 M^(p)（式 14）。"""
    n = len(Y['ids']); A = torch.zeros(n, n)
    if Y['edges']:
        e = torch.tensor(Y['edges']); A[e[:, 0], e[:, 1]] = 1; A[e[:, 1], e[:, 0]] = 1
    R = [A > 0]
    for _ in range(max(k_ctx, k_sim) - 1): R.append(R[-1] | ((R[-1].float() @ A) > 0))
    eye = torch.eye(n, dtype=torch.bool); R = [r & ~eye for r in R]
    comm = torch.tensor(km.predict(Y['x']))
    same = comm[:, None] == comm[None, :]
    d = torch.tensor(Y['deg'], dtype=torch.float32)
    ctx = R[k_ctx - 1] & same                                   # 式 2：同社区、k 跳可达
    Dm = torch.where(ctx, 1 / torch.sqrt(d[:, None].clamp(min=1) * d[None, :].clamp(min=1)), torch.zeros(()))
    prior = (xi * A + zeta * Dm).clamp(0, 1 - 1e-4)
    iu = torch.triu(prior > 0, diagonal=1).nonzero().T           # 候选点对（上三角）
    sims = []
    for k in range(k_sim):
        Rk = R[k].float(); inter = Rk @ Rk.T; sz = Rk.sum(1)
        uni = sz[:, None] + sz[None, :] - inter
        sims.append(torch.where(uni > 0, inter / uni.clamp(min=1), torch.zeros(()))[iu[0], iu[1]])
    T = A + torch.eye(n); T = T / T.sum(1, keepdim=True)        # 带自环的转移矩阵，孤立节点只转移到自己
    M, Tk = torch.zeros(n, n), torch.eye(n)
    for _ in range(p): Tk = Tk @ T; M = M + Tk
    M = M / p
    return dict(pairs=iu, prior=prior[iu[0], iu[1]], sim=torch.stack(sims, 1), D=Dm[iu[0], iu[1]], A_pair=A[iu[0], iu[1]],
                M=M, n=n, Amean=sparse_mean_adj(n, Y['edges']))

class StructAttn(nn.Module):
    """结构自注意力层（式 6–8、11–13）：注意力分数 = (Q·K)/√d_k + Φ·f_d(D_ij)，其中 Q、K 同时由节点表示 h 和点对的 k 跳相似度 s_ij 生成；
    在增强图的边（含自环）上做 softmax，乘以采样得到的边门控，再接残差、LayerNorm 和前馈网络。"""
    def __init__(s, d, heads, k_sim, drop=0.1):
        super().__init__()
        s.h, s.dk = heads, d // heads
        s.Wq, s.Wk, s.Wv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        s.Wqs, s.Wks = nn.Linear(k_sim, d), nn.Linear(k_sim, d)
        s.phi = nn.Linear(1, heads); s.O = nn.Linear(d, d)
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(drop), nn.Linear(2 * d, d))
    def forward(s, h, src, dst, sim, Dij, gate):
        n = h.shape[0]
        q = (s.Wq(h)[dst] + s.Wqs(sim)).view(-1, s.h, s.dk)     # 目标节点 i（dst）的查询
        k = (s.Wk(h)[src] + s.Wks(sim)).view(-1, s.h, s.dk)     # 邻居 j（src）的键
        v = s.Wv(h)[src].view(-1, s.h, s.dk)
        sc = (q * k).sum(-1) / s.dk ** .5 + s.phi(Dij[:, None])  # [边, 头]
        mx = torch.full((n, s.h), -1e9).scatter_reduce(0, dst[:, None].expand(-1, s.h), sc, 'amax', include_self=True)
        w = torch.exp(sc - mx[dst]) * gate[:, None]
        z = torch.zeros(n, s.h).index_add(0, dst, w) + 1e-9
        out = torch.zeros(n, s.h, s.dk).index_add(0, dst, w[:, :, None] * v) / z[:, :, None]
        h = s.ln1(h + s.O(out.reshape(n, -1)))
        return s.ln2(h + s.ffn(h))

class DFGT(nn.Module):
    """DegFairGT 式链路预测器：可学习结构增强（式 1、4，松弛伯努利 + 直通估计）→ 结构自注意力 Transformer → 节点表示；
    自监督损失 L1（重建 p 步转移概率和特征，式 14）+ α·L2（增强图与原图的 BCE，式 15），与链路损失联合训练（权重 λ）。"""
    def __init__(s, fin, hid=64, layers=2, heads=4, k_sim=3, beta1=.5, beta2=.5, alpha=1.):
        super().__init__()
        s.inp = nn.Linear(fin, hid); s.aug = nn.Sequential(nn.Linear(fin, hid), nn.ReLU(), nn.Linear(hid, hid))
        s.layers = nn.ModuleList(StructAttn(hid, heads, k_sim) for _ in range(layers))
        s.recx = nn.Sequential(nn.Linear(hid, 2 * hid), nn.SELU(), nn.Linear(2 * hid, fin))
        s.dec = PairDecoder(hid); s.b1, s.b2, s.alpha, s.hid = beta1, beta2, alpha, hid
    def gates(s, x, P):
        """式 1、4：对每个候选点对，logit = logit(先验) + tanh(学到的相似度)；训练时松弛伯努利采样（τ=1，直通），评估时取概率 > 0.5。"""
        z = s.aug(x + torch.sparse.mm(P['Amean'], x))           # 一层均值聚合 + MLP，得到用于打分的表示
        i, j = P['pairs']
        logit = torch.logit(P['prior']) + torch.tanh((z[i] * z[j]).sum(-1) / s.hid ** .5)
        prob = torch.sigmoid(logit)
        if s.training:
            u = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
            soft = torch.sigmoid(logit + torch.log(u) - torch.log(1 - u)); hard = (soft > .5).float()
            g = hard - soft.detach() + soft
        else:
            g = (prob > .5).float()
        return g, prob
    def embed(s, x, P):
        n = P['n']; i, j = P['pairs']; g, prob = s.gates(x, P)
        keep = g > 0 if not s.training else torch.ones_like(g, dtype=torch.bool)
        ar = torch.arange(n)
        src = torch.cat([i[keep], j[keep], ar]); dst = torch.cat([j[keep], i[keep], ar])
        sim = torch.cat([P['sim'][keep], P['sim'][keep], torch.ones(n, P['sim'].shape[1])])
        Dij = torch.cat([P['D'][keep], P['D'][keep], torch.zeros(n)]); gate = torch.cat([g[keep], g[keep], torch.ones(n)])
        h = s.inp(x)
        for layer in s.layers: h = layer(h, src, dst, sim, Dij, gate)
        if not s.training: return h, torch.zeros(())
        zn = F.normalize(h, dim=1)
        L1 = s.b1 * ((P['M'] - zn @ zn.T) ** 2).mean() + s.b2 * ((x - s.recx(h)) ** 2).mean()
        L2 = F.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), P['A_pair'])
        return h, L1 + s.alpha * L2

def fit_kmeans(Ys, years=(2021, 2022), M=5, seed=0):
    """在训练年节点特征上拟合 k-means（M=5 个社区），各年用同一个模型划分社区。"""
    return KMeans(n_clusters=M, n_init=10, random_state=seed).fit(np.vstack([Ys[t]['x'] for t in years]))
