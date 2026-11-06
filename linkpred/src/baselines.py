"""TNSE 实验 B 的新基线（DEV_RESULTS 第 16.2 节）。全部按论文重写，**不是官方实现**；只重写各方法的核心机制，
统一使用本项目的训练信号（“新关系”标签上的 BCE）、早停和评估。每个模型提供 prepare(Y) → P、embed(x, P) → (h, 辅助损失)、dec(h, u, v)。
- GAE（Kipf & Welling 2016）：2 层 GCN（对称归一化、带自环），内积解码器 σ(h_u·h_v)。
- Cold Brew（Zheng et al. 2022）：教师 = GraphSAGE 式 GNN；学生 = 只用节点特征的 MLP，用 MSE 蒸馏教师表示（只在度 > 0 的节点上）；
  最终表示：度 > 0 用教师，度 = 0（孤立）用学生。原文是两阶段训练，这里简化为一阶段联合训练（辅助损失 = 蒸馏 MSE）。
- DegFairGNN（Liu et al. 2023）：按度数调制的去偏上下文：聚合后的邻居信息加上 σ(w·log(1+d))⊙c_low − (1−σ(·))⊙c_high 形式的可学习补偿
  （低度节点获得补偿，高度节点去掉多余信息）；辅助损失 = 低度组与高度组平均链路损失之差的平方（度阈值 = 1，与本文分组一致）。
- Tail-GNN（Liu, Nguyen & Fang 2021）：邻域翻译：r_i = MLP([h_i, m_i])，预测“缺失的邻居信息”；对度 ≥ K_HEAD 的头部节点，随机只保留 1 个邻居
  构造伪尾部，训练 r 恢复完整邻居均值（辅助损失 = MSE）；聚合时对度 < K_HEAD 的节点加上 r_i。原文的判别器省略。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from paper_models import PairDecoder, sparse_mean_adj

def sym_adj(n, edges):
    """GCN 的对称归一化邻接 D^{-1/2}(A+I)D^{-1/2}。"""
    e = np.array(edges, dtype=np.int64).reshape(-1, 2)
    r = np.concatenate([e[:, 0], e[:, 1], np.arange(n)]); c = np.concatenate([e[:, 1], e[:, 0], np.arange(n)])
    d = np.bincount(r, minlength=n).astype(np.float32); w = 1 / np.sqrt(d[r] * d[c])
    return torch.sparse_coo_tensor(torch.tensor(np.vstack([r, c])), torch.tensor(w, dtype=torch.float32), (n, n)).coalesce()

class GAE(nn.Module):
    def __init__(s, fin, hid=128, drop=0.3, **kw):
        super().__init__(); s.l1, s.l2 = nn.Linear(fin, hid), nn.Linear(hid, hid); s.drop = nn.Dropout(drop)
    def prepare(s, Y): return dict(A=sym_adj(len(Y['ids']), Y['edges']))
    def embed(s, x, P):
        h = s.drop(torch.relu(torch.sparse.mm(P['A'], s.l1(x)))); return torch.sparse.mm(P['A'], s.l2(h)), torch.zeros(())
    def dec(s, h, u, v): return (h[u] * h[v]).sum(-1) / h.shape[1] ** .5

class SageEnc(nn.Module):
    """2 层均值聚合编码器（与 paper_models.SAGE 相同的结构），供 Cold Brew 的教师、DegFairGNN 和 Tail-GNN 共用。"""
    def __init__(s, fin, hid, drop):
        super().__init__(); s.self_l = nn.ModuleList([nn.Linear(fin, hid), nn.Linear(hid, hid)]); s.nb_l = nn.ModuleList([nn.Linear(fin, hid), nn.Linear(hid, hid)]); s.drop = nn.Dropout(drop)

class ColdBrew(nn.Module):
    def __init__(s, fin, hid=128, drop=0.3, **kw):
        super().__init__(); s.t = SageEnc(fin, hid, drop); s.stu = nn.Sequential(nn.Linear(fin, hid), nn.ReLU(), nn.Linear(hid, hid)); s.dec_ = PairDecoder(hid)
    def prepare(s, Y): return dict(A=sparse_mean_adj(len(Y['ids']), Y['edges']), has=torch.tensor(np.asarray(Y['deg']) > 0))
    def embed(s, x, P):
        h = x
        for k, (a, b) in enumerate(zip(s.t.self_l, s.t.nb_l)):
            h = a(h) + b(torch.sparse.mm(P['A'], h))
            if k == 0: h = s.t.drop(torch.relu(h))
        z = s.stu(x); has = P['has']
        kd = F.mse_loss(z[has], h[has].detach()) if has.any() else torch.zeros(())
        return torch.where(has[:, None], h, z), kd
    def dec(s, h, u, v): return s.dec_(h, u, v)

class DegFair(nn.Module):
    def __init__(s, fin, hid=128, drop=0.3, **kw):
        super().__init__(); s.e = SageEnc(fin, hid, drop); s.c_lo = nn.Parameter(torch.zeros(2, hid)); s.c_hi = nn.Parameter(torch.zeros(2, hid))
        s.w = nn.Parameter(torch.ones(2)); s.b = nn.Parameter(torch.zeros(2)); s.dec_ = PairDecoder(hid); s.fair = True
    def prepare(s, Y): return dict(A=sparse_mean_adj(len(Y['ids']), Y['edges']), ld=torch.log1p(torch.tensor(np.asarray(Y['deg']), dtype=torch.float32)),
                                   deg=torch.tensor(np.asarray(Y['deg']), dtype=torch.float32))
    def embed(s, x, P):
        h = x
        for k, (a, b) in enumerate(zip(s.e.self_l, s.e.nb_l)):
            m = b(torch.sparse.mm(P['A'], h)); g = torch.sigmoid(s.w[k] * P['ld'] + s.b[k])[:, None]
            h = a(h) + m + (1 - g) * s.c_lo[k] - g * s.c_hi[k]        # 低度节点（g 小）获得补偿 c_lo，高度节点去掉 c_hi
            if k == 0: h = s.e.drop(torch.relu(h))
        return h, torch.zeros(())
    def dec(s, h, u, v): return s.dec_(h, u, v)
    def fair_loss(s, per_pair_loss, P, u, v):
        """低度组（较小端度 ≤ 1）与高度组平均链路损失之差的平方。"""
        lo = torch.minimum(P['deg'][u], P['deg'][v]) <= 1
        if lo.all() or (~lo).all(): return torch.zeros(())
        return (per_pair_loss[lo].mean() - per_pair_loss[~lo].mean()) ** 2

K_HEAD = 3
class TailGNN(nn.Module):
    def __init__(s, fin, hid=128, drop=0.3, **kw):
        super().__init__(); s.e = SageEnc(fin, hid, drop); s.tr = nn.ModuleList(nn.Sequential(nn.Linear(2 * d, hid), nn.ReLU(), nn.Linear(hid, d)) for d in (fin, hid)); s.dec_ = PairDecoder(hid)
    def prepare(s, Y):
        n = len(Y['ids']); deg = np.asarray(Y['deg']); e = np.array(Y['edges'], dtype=np.int64).reshape(-1, 2)
        rng = np.random.default_rng(0); keep = {}                    # 伪尾部：每个头部节点只保留 1 条随机边
        for i, j in e:
            for a, b in ((i, j), (j, i)):
                if deg[a] >= K_HEAD and (a not in keep or rng.random() < 1 / deg[a]): keep[a] = b
        pe = [(a, b) for a, b in keep.items()]
        Ap = torch.sparse_coo_tensor(torch.tensor(np.array(pe, dtype=np.int64).T.reshape(2, -1)), torch.ones(len(pe)), (n, n)).coalesce() if pe else None
        return dict(A=sparse_mean_adj(n, Y['edges']), Ap=Ap, head=torch.tensor(deg >= K_HEAD), tail=torch.tensor(deg < K_HEAD))
    def embed(s, x, P):
        h = x; aux = torch.zeros(())
        for k, (a, b) in enumerate(zip(s.e.self_l, s.e.nb_l)):
            m = torch.sparse.mm(P['A'], h); r = s.tr[k](torch.cat([h, m], -1))
            if P['Ap'] is not None and P['head'].any():               # 头部节点：从伪尾部邻居恢复完整邻居均值
                mp = torch.sparse.mm(P['Ap'], h); rp = s.tr[k](torch.cat([h, mp], -1))
                aux = aux + F.mse_loss((mp + rp)[P['head']], m[P['head']].detach())
            m = m + P['tail'][:, None] * r
            h = a(h) + b(m)
            if k == 0: h = s.e.drop(torch.relu(h))
        return h, aux
    def dec(s, h, u, v): return s.dec_(h, u, v)

BASELINES = {'gae': GAE, 'coldbrew': ColdBrew, 'degfair': DegFair, 'tailgnn': TailGNN}
BASE_GRID = [dict(hid=h, lr=lr) for h in (64, 128) for lr in (1e-3, 1e-2)]   # 每个基线 4 个配置，只在 F1、F2 上选（第 16.2 节）
