"""DEV_RESULTS 第 29 节：论文中几个数字的复核（只读已有数据；不产生新分数）。
(1) 验证年 575 个正例 vs 全部开发年 989 个正例：兼任 / 大股东信号占比；
(2) 标准基准（Cora、CiteSeer、Photo、Computers）的孤立节点占比；
(3) 验证年中 Adamic–Adar 在有孤立端点的正例上的组内 AUC（按年）。
用法：python src/check_paper_numbers.py"""
import collections, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
L = Path(__file__).resolve().parents[1]

# (1) 信号占比
X = pd.read_csv(L / 'data_v3x/pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str}, usecols=['pair_id', 'fy', 'anchor', 'other', 'label'])
IL = pd.read_csv(L / 'out_dev_rewire/real/interlock_feats.csv', index_col='row').reindex(X.index)
HS = pd.read_csv(L / 'out_dev_rewire/real/hyslr_feats.csv', index_col='row').reindex(X.index)
hcols = [c for c in HS.columns if c.startswith('hs_') and c not in ('hs_has_a', 'hs_has_o')]
S = {'direct': IL.il_direct > 0, 'any_interlock': (IL[['il_direct', 'il_inside', 'il_2hop', 'il_hist']] > 0).any(axis=1), 'any_shareholder': (HS[hcols].fillna(0) > 0).any(axis=1)}
S['either'] = S['any_interlock'] | S['any_shareholder']
for name, mask in (('验证年 2021–2023', X.fy.between(2021, 2023)), ('全部开发年 2019–2023', X.fy.between(2019, 2023))):
    P, N = X.index[mask & (X.label == 1)], X.index[mask & (X.label == 0)]
    print(f'{name}: 正例 {len(P)}，负例 {len(N)}')
    for k, s in S.items(): print(f'   {k:16s} 正例 {s[P].mean():.4f}  负例 {s[N].mean():.5f}')

# (2) 基准的孤立节点占比（用已下载并处理好的数据；root 为 bench/raw，数据集名为子目录名）
try:
    import torch
    from torch_geometric.datasets import Planetoid, Amazon
    R = str(L / 'bench/raw')
    for name, ds in (('Cora', lambda: Planetoid(R, 'Cora')), ('CiteSeer', lambda: Planetoid(R, 'CiteSeer')), ('Photo', lambda: Amazon(R, 'Photo')), ('Computers', lambda: Amazon(R, 'Computers'))):
        d = ds()[0]; ei = d.edge_index; deg = torch.zeros(d.num_nodes, dtype=torch.long)
        deg.scatter_add_(0, ei[0], torch.ones(ei.size(1), dtype=torch.long)); deg.scatter_add_(0, ei[1], torch.ones(ei.size(1), dtype=torch.long))
        iso = int((deg == 0).sum()); print(f'{name}: 节点 {d.num_nodes}，度为 0 的节点 {iso}（{100 * iso / d.num_nodes:.2f}%）')
except Exception as e:
    print('基准检查未运行：', e)

# (3) AA 在有孤立端点的正例上的组内 AUC
for t in (2021, 2022, 2023):
    g = pd.read_csv(L / f'data_v3x/graph_{t}.csv', dtype=str)
    nb = collections.defaultdict(set); dcnt = collections.Counter(list(g.src) + list(g.dst))
    for a, b in zip(g.src, g.dst):
        if a != b: nb[a].add(b); nb[b].add(a)
    aa = lambda u, v: sum(1 / math.log(len(nb[z])) for z in nb[u] & nb[v] if len(nb[z]) > 1)
    allv, anc, oth = [], [], []
    for pid, grp in X[X.fy == t].groupby('pair_id'):
        p = grp[grp.label == 1].iloc[0]; neg = grp[grp.label == 0]
        if min(dcnt[p.anchor], dcnt[p.other]) != 0: continue
        sp = aa(p.anchor, p.other); sn = np.array([aa(p.anchor, o) for o in neg.other])
        auc = ((sn < sp).sum() + .5 * (sn == sp).sum()) / len(sn); allv.append(auc); (anc if dcnt[p.anchor] == 0 else oth).append(auc)
    print(f'AA {t}: 有孤立端点的正例 {len(allv)}，组内 AUC {np.mean(allv):.4f}（锚点孤立 n={len(anc)}：{np.mean(anc):.4f}；只有对方孤立 n={len(oth)}：{np.mean(oth):.4f}）')
