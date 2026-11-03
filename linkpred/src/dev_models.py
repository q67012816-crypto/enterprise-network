"""开发阶段（只用训练年和验证年）的模型扩展：在 paper_models.py（A7 冻结版本，不修改）的基础上增加
  1) 消融开关：关闭结构增强（aug=False）、关闭结构相似度项（use_sim=False）；自监督由 λ=0 关闭；
  2) 零度上下文（DFGT-0）：度为 0 的节点没有 k 跳邻居，原方法覆盖不到。这里为每个零度节点在同一 k-means 社区里
     按特征余弦相似度取最相似的 k 个节点作为候选补边（先验 = ζ·D_ij，D_ij = 1/√(max(d_i,1)·max(d_j,1))）。"""
import numpy as np, torch, torch.nn.functional as F
from paper_models import DFGT, dfgt_prepare

def dfgt_prepare_dev(Y, km, zero_k=0, zeta=0.2, **kw):
    """在原有预计算（式 1–3、9–10、14）之上，可选地为零度节点加入特征相似的同社区候选点对。
    输入：节点表 Y、k-means、zero_k（每个零度节点补几个候选，0 表示不补）；输出：与 dfgt_prepare 相同结构的 dict。"""
    P = dfgt_prepare(Y, km, zeta=zeta, **kw)
    if zero_k <= 0: return P
    x = torch.tensor(Y['x']); deg = torch.tensor(Y['deg'], dtype=torch.float32); n = len(deg)
    comm = torch.tensor(km.predict(Y['x'])); xn = F.normalize(x, dim=1)
    zero = torch.nonzero(deg == 0).flatten()
    have = set(map(tuple, P['pairs'].T.tolist())); add = []
    for i in zero.tolist():
        same = torch.nonzero((comm == comm[i]) & (torch.arange(n) != i)).flatten()
        if len(same) == 0: continue
        s = xn[same] @ xn[i]; top = same[torch.topk(s, min(zero_k, len(same))).indices]
        for j in top.tolist():
            a, b = min(i, j), max(i, j)
            if (a, b) not in have: have.add((a, b)); add.append((a, b))
    if not add: return P
    ad = torch.tensor(add).T; Dn = 1 / torch.sqrt(deg[ad[0]].clamp(min=1) * deg[ad[1]].clamp(min=1))
    P = dict(P); P['pairs'] = torch.cat([P['pairs'], ad], 1); P['prior'] = torch.cat([P['prior'], (zeta * Dn).clamp(1e-4, 1 - 1e-4)])
    P['sim'] = torch.cat([P['sim'], torch.zeros(ad.shape[1], P['sim'].shape[1])]); P['D'] = torch.cat([P['D'], Dn]); P['A_pair'] = torch.cat([P['A_pair'], torch.zeros(ad.shape[1])])
    P['n_zero_added'] = ad.shape[1]
    return P

class DFGTDev(DFGT):
    """DFGT 加消融开关。aug=False：不做结构增强，只在原图的边（加自环）上做注意力，门控恒为 1；
    use_sim=False：注意力里去掉 k 跳相似度 s_ij 和 D_ij 项（式 6–8 退化为普通图注意力）。"""
    def __init__(s, fin, aug=True, use_sim=True, soft_eval=False, **kw):
        super().__init__(fin, **kw); s.use_aug, s.use_sim, s.soft_eval = aug, use_sim, soft_eval
    def _P(s, P):
        if s.use_aug and s.use_sim: return P
        Q = dict(P)
        if not s.use_sim: Q['sim'] = torch.zeros_like(P['sim']); Q['D'] = torch.zeros_like(P['D'])
        if not s.use_aug:
            m = P['A_pair'] > 0
            for k in ('pairs',): Q[k] = P[k][:, m]
            for k in ('prior', 'sim', 'D', 'A_pair'): Q[k] = Q[k][m]
        return Q
    def gates(s, x, P):
        """soft_eval=True 时，评估阶段用期望门控（每条候选边以其概率加权），而不是"概率 > 0.5 才保留"。
        原因：增强边的先验最高 ζ·D ≤ 0.2，加上 tanh 项后概率仍 < 0.5，硬阈值会把所有增强边在预测时丢掉（A7 版本的缺陷）。"""
        if s.use_aug and s.soft_eval and not s.training:
            _, prob = super().gates(x, P); return prob, prob
        if s.use_aug: return super().gates(x, P)
        g = torch.ones(P['pairs'].shape[1]); return g, g.clamp(1e-6, 1 - 1e-6)
    def embed(s, x, P):
        return super().embed(x, s._P(P))
