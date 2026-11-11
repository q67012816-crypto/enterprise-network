"""“结构 + 点对证据”双通道模型（DCM，DEV_RESULTS 第 14 节；开发阶段）。纯 PyTorch。
A. 关系类型分开聚合（R-GCN 式）：持股边、兼任边各一套权重做均值消息传递，用节点层面的门控合并；
   rel=False（消融 A）时两类边合并成一张图，同一套权重。
B. 按度数门控的点对证据：e = MLP(点对特征)；α = σ(MLP(log1p(min_deg), log1p(max_deg)))；修正量 = α·e + (1 − α)·s_结构；
   gate=False（消融 B）时修正量 = e + s_结构。
C. 残差学习：最终 logit = 基础分数（逻辑回归 M1 + il 的 logit）+ 修正量；resid=False（消融 C）时不加基础分数。"""
import torch, torch.nn as nn
from paper_models import PairDecoder, sparse_mean_adj

def dcm_prepare(n, own_edges, il_edges):
    """每年一次：持股图、兼任图、合并图的行归一化稀疏邻接矩阵（孤立节点那一行为 0）。"""
    return dict(A_own=sparse_mean_adj(n, own_edges), A_il=sparse_mean_adj(n, il_edges), A_all=sparse_mean_adj(n, sorted(set(own_edges) | set(il_edges))))

class DualChannel(nn.Module):
    def __init__(s, fin, fpair, hid=64, layers=2, drop=0.3, rel=True, gate=True, resid=True):
        super().__init__(); s.rel, s.gate, s.resid = rel, gate, resid
        s.inp = nn.Linear(fin, hid); s.drop = nn.Dropout(drop)
        s.W0 = nn.ModuleList(nn.Linear(hid, hid) for _ in range(layers))
        if rel:
            s.Wo = nn.ModuleList(nn.Linear(hid, hid, bias=False) for _ in range(layers)); s.Wi = nn.ModuleList(nn.Linear(hid, hid, bias=False) for _ in range(layers))
            s.go = nn.ModuleList(nn.Linear(2 * hid, 1) for _ in range(layers)); s.gi = nn.ModuleList(nn.Linear(2 * hid, 1) for _ in range(layers))
        else:
            s.Wa = nn.ModuleList(nn.Linear(hid, hid, bias=False) for _ in range(layers))
        s.dec = PairDecoder(hid)
        s.ev = nn.Sequential(nn.Linear(fpair, 32), nn.ReLU(), nn.Linear(32, 1))
        s.g = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))

    def embed(s, x, P):
        h = torch.relu(s.inp(x))
        for l in range(len(s.W0)):
            if s.rel:
                mo, mi = torch.sparse.mm(P['A_own'], h), torch.sparse.mm(P['A_il'], h)
                go = torch.sigmoid(s.go[l](torch.cat([h, mo], -1))); gi = torch.sigmoid(s.gi[l](torch.cat([h, mi], -1)))
                h = torch.relu(s.W0[l](h) + go * s.Wo[l](mo) + gi * s.Wi[l](mi))
            else:
                h = torch.relu(s.W0[l](h) + s.Wa[l](torch.sparse.mm(P['A_all'], h)))
            h = s.drop(h)
        return h

    def logit(s, h, u, v, pf, dg, base):
        """pf：点对特征；dg：[log1p(min_deg), log1p(max_deg)]（持股图）；base：逻辑回归 logit。"""
        st = s.dec(h, u, v); e = s.ev(pf).squeeze(-1)
        if s.gate:
            a = torch.sigmoid(s.g(dg)).squeeze(-1); c = a * e + (1 - a) * st
        else: c = e + st
        return base + c if s.resid else c
