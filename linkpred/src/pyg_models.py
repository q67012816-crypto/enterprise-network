"""DEV_RESULTS 第 20.2 节：PyG 官方层实现的 GCN、GAT（都加自环）及其无图对照（同一网络、去掉所有边，只剩自环）。
接口与 paper_models.SAGE 相同：prepare(Y) → 结构输入；embed(x, P) → (节点表示, 0)；dec = 同一个 MLP 解码器。"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from paper_models import PairDecoder

class PygGNN(nn.Module):
    def __init__(s, fin, hid=128, kind='gcn', nograph=False, drop=0.3):
        super().__init__(); s.kind, s.nograph = kind, nograph
        if kind == 'gcn': s.c1, s.c2 = GCNConv(fin, hid, add_self_loops=True), GCNConv(hid, hid, add_self_loops=True)
        else: s.c1, s.c2 = GATConv(fin, hid // 4, heads=4, add_self_loops=True), GATConv(hid, hid, heads=1, add_self_loops=True)
        s.drop = nn.Dropout(drop); s.dec = PairDecoder(hid)
    def prepare(s, Y):
        e = [] if s.nograph else Y['edges']
        ei = torch.tensor(e, dtype=torch.long).t() if len(e) else torch.zeros(2, 0, dtype=torch.long)
        return dict(ei=torch.cat([ei, ei.flip(0)], 1))
    def embed(s, x, P):
        act = F.relu if s.kind == 'gcn' else F.elu
        h = s.drop(act(s.c1(x, P['ei']))); return s.c2(h, P['ei']), torch.zeros(())
