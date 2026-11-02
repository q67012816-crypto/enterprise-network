"""M3：GraphSAGE 链接预测（protocol.md §7 + A6；protocol_confirm 主指标=组内 AUC）。
探索性：2024→2025 测试年已被查看过，所有测试数字只能作探索性/相关性解读。

用法：
  python3 src/m3_sage.py search          # 网格 × 5 种子，只用训练集训练、验证集早停与选配置（不读测试标签）
  python3 src/m3_sage.py test            # 用选定配置的 5 个种子模型给测试集打分，只能运行一次（输出已存在则拒绝）
"""
import itertools, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import average_precision_score as AP

L = Path(__file__).resolve().parents[1]
D = L / 'data_a2_apionly'          # A4 方案 a：所有年份只用 API 规则建图
O = L / 'out_2_m3'; O.mkdir(exist_ok=True); (O / 'models').mkdir(exist_ok=True)
torch.set_num_threads(8)

# ---------- 1. 读取数据（与 models_v2.py 相同的文件与特征口径） ----------
X = pd.read_csv(D / 'pairs.csv', dtype={'anchor': str, 'other': str})
N = pd.read_csv(D / 'nodes.csv', dtype={'corp_code': str, 'industry': str, 'market': str, 'fin_disc': str}, keep_default_na=False, na_values=[''])
G = {t: pd.read_csv(D / f'graph_{t}.csv', dtype=str) for t in range(2021, 2025)}
FI = ['log_assets', 'liabilities_assets', 'equity_assets', 'capital_assets', 'revenue_assets', 'operating_income_assets',
      'net_income_assets', 'signed_log_revenue', 'signed_log_operating_income', 'current_operating_loss']
# 预处理统计量只用训练年份（2021、2022）的节点，和 M1 完全一致：0.5/99.5 分位截断、中位数填补 + 缺失指示
tr_nodes = N[N.fy.isin([2021, 2022])]
lo, hi = tr_nodes[FI].quantile(.005), tr_nodes[FI].quantile(.995)
med = tr_nodes[FI].clip(lo, hi, axis=1).median()
topind = tr_nodes.industry.value_counts().index[:20].tolist()

def node_table(t):
    """年份 t 的节点集合与特征矩阵：节点=该年合格企业 ∪ G_t 中出现的企业。"""
    Nt = N[N.fy == t].set_index('corp_code')
    ids = sorted(set(Nt.index) | set(G[t].src) | set(G[t].dst))
    M = Nt.reindex(ids); cols = []
    for c in FI:
        cols.append(M[c].isna().astype(float).values)                       # 缺失指示
        cols.append(M[c].clip(lo[c], hi[c]).fillna(med[c]).values)          # 截断 + 中位数填补
    ind = M.industry.fillna('').values; mk = M.market.fillna('').values
    for i in topind: cols.append((ind == i).astype(float))                  # 行业（KSIC 前两位）one-hot，前 20 + 其他
    cols.append((~np.isin(ind, topind)).astype(float))
    for m in ('Y', 'K'): cols.append((mk == m).astype(float))               # 市场（当前值，A3）
    F = np.vstack(cols).T
    return ids, F

YEARS = {}
mu = sd = None
for t in range(2021, 2025):
    ids, F = node_table(t); YEARS[t] = dict(ids=ids, idx={c: i for i, c in enumerate(ids)}, F=F)
tr_F = np.vstack([YEARS[t]['F'] for t in (2021, 2022)])
mu, sd = tr_F.mean(0), tr_F.std(0); sd[sd == 0] = 1                         # 标准化同样只用训练年份
for t, Y in YEARS.items():
    Y['x'] = torch.tensor((Y['F'] - mu) / sd, dtype=torch.float32)
    # 无向邻接（行归一化 = 均值聚合）；消息传递只用当年的 G_t，不含 t+1 的任何边
    s = [Y['idx'][a] for a in G[t].src]; d = [Y['idx'][b] for b in G[t].dst]
    e = set(zip(s, d)) | set(zip(d, s)); e = {p for p in e if p[0] != p[1]}
    r, c = map(list, zip(*e)) if e else ([], [])
    n = len(Y['ids']); deg = np.bincount(r, minlength=n).astype(float)
    val = 1 / deg[r] if e else []
    Y['A'] = torch.sparse_coo_tensor([r, c], torch.tensor(val, dtype=torch.float32), (n, n)).coalesce()   # 孤立节点的邻居均值=0

def pair_index(df):
    """按年份把 (anchor, other) 映射成节点下标。"""
    out = {}
    for t, g in df.groupby('fy'):
        Y = YEARS[t]; out[t] = (torch.tensor([Y['idx'][a] for a in g.anchor]), torch.tensor([Y['idx'][o] for o in g.other]), g.index.values)
    return out
TR, VA = X[X.split == 'train'], X[X.split == 'validation']
P_TR, P_VA = pair_index(TR), pair_index(VA)
y_tr = torch.tensor(X.label.values, dtype=torch.float32)

# ---------- 2. 模型：GraphSAGE（均值聚合）+ 点积或 MLP 解码器 ----------
class SAGE(nn.Module):
    def __init__(s, fin, hid, layers, drop, dec):
        super().__init__()
        dims = [fin] + [hid] * layers
        s.self_l = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))     # 自身表示
        s.nb_l = nn.ModuleList(nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(layers))  # 邻居均值表示
        s.drop, s.dec = nn.Dropout(drop), dec
        if dec == 'mlp': s.mlp = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, 1))
    def embed(s, x, A):
        h = x
        for k, (a, b) in enumerate(zip(s.self_l, s.nb_l)):
            h = a(h) + b(torch.sparse.mm(A, h))    # h_v' = W1·h_v + W2·mean(h_u, u∈N(v))
            if k < len(s.self_l) - 1: h = s.drop(torch.relu(h))
        return h
    def score(s, h, u, v):
        if s.dec == 'dot': return (h[u] * h[v]).sum(-1)
        return s.mlp(torch.cat([h[u] * h[v], (h[u] - h[v]).abs()], -1)).squeeze(-1)   # 对称的 MLP 解码器

def predict(m, P):
    """对一组按年分好的点对打分（eval 模式，无 dropout）。"""
    m.eval(); out = np.zeros(len(X))
    with torch.no_grad():
        for t, (u, v, ix) in P.items(): out[ix] = torch.sigmoid(m.score(m.embed(YEARS[t]['x'], YEARS[t]['A']), u, v)).numpy()
    return out

def within_auc(df, s):
    """组内 AUC：每个正例在自己的 101 个候选（同 pair_id）中排序，并列取期望秩（A1/A5）。"""
    d = pd.DataFrame({'pid': df.pair_id.values, 'y': df.label.values, 's': s}); r = []
    for _, g in d.groupby('pid', sort=False):
        x = g.s[g.y == 1].iloc[0]; ns = g.s[g.y == 0].values
        r.append(((ns < x).sum() + (ns == x).sum() / 2) / len(ns))
    return float(np.mean(r))

def train_one(cfg, seed):
    hid, layers, lr, drop, dec = cfg
    torch.manual_seed(seed); np.random.seed(seed)
    m = SAGE(YEARS[2021]['x'].shape[1], hid, layers, drop, dec); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=0)
    lossf = nn.BCEWithLogitsLoss(); best, best_state, bad, ep_best = -1, None, 0, 0
    yv = VA.label.values
    for ep in range(1, 201):                       # 最多 200 轮
        m.train(); opt.zero_grad(); loss = 0
        for t, (u, v, ix) in P_TR.items():         # 训练集：2021→22、2022→23 的正例与已抽样负例（全批量）
            loss = loss + lossf(m.score(m.embed(YEARS[t]['x'], YEARS[t]['A']), u, v), y_tr[ix]) * len(ix)
        (loss / len(TR)).backward(); opt.step()
        sv = predict(m, P_VA)[VA.index]; a = AP(yv, sv)   # 早停：验证 AP，patience 20（§7 原文）
        if a > best: best, bad, ep_best, best_state = a, 0, ep, {k: w.clone() for k, w in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 20: break
    m.load_state_dict(best_state)
    sv = predict(m, P_VA)[VA.index]
    return m, dict(val_AP=AP(yv, sv), val_within_AUC=within_auc(VA, sv), best_epoch=ep_best, epochs_run=ep), sv

GRID = list(itertools.product([64, 128], [1, 2], [1e-3, 1e-2], [0.0, 0.3], ['dot', 'mlp']))
SEEDS = [1, 2, 3, 4, 5]
cname = lambda c: f'h{c[0]}_l{c[1]}_lr{c[2]}_do{c[3]}_{c[4]}'

if sys.argv[1] == 'search':
    rows = []; t0 = time.time()
    for cfg in GRID:
        for sd_ in SEEDS:
            m, r, sv = train_one(cfg, sd_)
            torch.save(m.state_dict(), O / 'models' / f'{cname(cfg)}_s{sd_}.pt')
            rows.append(dict(config=cname(cfg), seed=sd_, **r)); print(rows[-1], f'{time.time()-t0:.0f}s', flush=True)
    R = pd.DataFrame(rows); R.to_csv(O / 'val_grid_runs.csv', index=False)
    S = R.groupby('config')[['val_within_AUC', 'val_AP']].mean().sort_values('val_within_AUC', ascending=False)
    S.to_csv(O / 'val_grid_summary.csv')
    best = S.index[0]   # A6：按 5 种子平均验证组内 AUC 选配置
    json.dump(dict(selected_config=best, rule='mean validation within-group AUC over seeds 1-5 (protocol_confirm; A6)',
                   val=S.loc[best].to_dict(), selected_by_val_AP_for_reference=S.val_AP.idxmax()), open(O / 'selection.json', 'w'), indent=1)
    # 保存选定配置的验证分数（5 种子平均），供验证集分组诊断
    print('selected', best)

elif sys.argv[1] == 'test':
    outf = O / 'test_scores_m3.csv'
    if outf.exists(): sys.exit('测试已评估过一次，拒绝重复运行（protocol_confirm 测试评估规则）')
    sel = json.load(open(O / 'selection.json'))['selected_config']
    hid, layers, lr, drop, dec = [p for p in sel.split('_')]
    cfg = (int(hid[1:]), int(layers[1:]), float(lr[2:]), float(drop[2:]), dec)
    TE = X[X.split == 'test']; P_TE = pair_index(TE); S = []
    for sd_ in SEEDS:
        m = SAGE(YEARS[2021]['x'].shape[1], *cfg[:2], cfg[3], cfg[4]); m.load_state_dict(torch.load(O / 'models' / f'{sel}_s{sd_}.pt'))
        S.append(predict(m, P_TE)[TE.index])
    out = TE[['pair_id', 'anchor', 'other', 'label']].copy(); out['M3'] = np.mean(S, 0)    # 5 种子预测取平均
    for i, s in enumerate(S): out[f'M3_s{i+1}'] = s
    out.to_csv(outf, index=False)
    json.dump(dict(test_evaluated_once=True, time=time.strftime('%Y-%m-%d %H:%M:%S KST'), config=sel), open(O / 'TEST_DONE.json', 'w'))
    print('测试集打分完成（仅此一次）', sel)
