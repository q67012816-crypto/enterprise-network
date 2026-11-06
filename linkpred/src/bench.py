"""TNSE 实验 A：受控孤立节点实验（检验命题 2；DEV_RESULTS 第 16 节，设计和判定标准在实验前提交，提交 31937b5）。
公开基准（PyG 2.8.1 标准版本，数据在 linkpred/bench/raw）：Cora、CiteSeer、Photo、Computers。
划分（数据种子 20261106）：无向边随机分为训练 85% / 验证 5% / 测试 10%（各 r 共用同一划分）；节点随机排列一次，
前 ⌊r·n⌋ 个为孤立节点（各 r 嵌套），删除它们在训练图中的全部边（既不传递消息，也不作训练正例）。
评估：每条验证 / 测试边随机取一端为锚，从与锚在完整图中不相邻的节点中抽 100 个负例；训练边每条配 20 个负例。
模型：M1（逻辑回归，节点特征先做 TruncatedSVD 到 128 维，再取 [h_u⊙h_v, |h_u−h_v|]；与第 16 节的唯一偏差，原因是原始维度下内存不够）、
GraphSAGE（hidden 128，lr 1e-2）、DFGT（lr 3e-3，λ 0.1）、无图对照（DFGT 架构，去掉所有边）；不针对新数据调参。
早停：验证组内 AUC，patience 15，最多 150 轮；测试分数取最佳验证轮。每个 GNN 10 个种子，同时保存每个种子的分数，并记录训练时间和 GPU 峰值显存。
用法：python src/bench.py run --data Cora --r 0.3 [--seeds 10]   → bench/out/<data>_r<r>.csv、bench/out/runtime.csv
      python src/bench.py eval                                    → bench/out/summary.json、summary.txt"""
import argparse, collections, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import within_metrics

L = Path(__file__).resolve().parents[1]; B = L / 'bench'; OUT = B / 'out'; OUT.mkdir(parents=True, exist_ok=True)
DATA = ('Cora', 'CiteSeer', 'Photo', 'Computers'); RS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5); SEED_DATA = 20261106; K_TR, K_EV = 20, 100

def load(name):
    from torch_geometric.datasets import Planetoid, Amazon
    d = (Planetoid(str(B / 'raw'), name) if name in ('Cora', 'CiteSeer') else Amazon(str(B / 'raw'), name))[0]
    e = d.edge_index.numpy(); e = e[:, e[0] != e[1]]; e = np.unique(np.sort(e, 0), axis=1).T   # 无向、去自环、去重
    x = d.x.numpy().astype(np.float32); x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1)
    return x, e

def split(n, E, r):
    """固定划分 + 嵌套的孤立节点集合；返回训练图的边、度，以及训练 / 验证 / 测试点对（每个正例后紧跟它的负例）。"""
    rng = np.random.default_rng(SEED_DATA); pe = rng.permutation(len(E)); n_tr, n_va = int(.85 * len(E)), int(.05 * len(E))
    tr, va, te = E[pe[:n_tr]], E[pe[n_tr:n_tr + n_va]], E[pe[n_tr + n_va:]]
    iso = np.zeros(n, bool); iso[rng.permutation(n)[:int(r * n)]] = True if r > 0 else False
    tr_g = tr[~(iso[tr[:, 0]] | iso[tr[:, 1]])]
    nb = collections.defaultdict(set)
    for a, b in E: nb[a].add(b); nb[b].add(a)
    def pairs(P, k, seed):
        g = np.random.default_rng(seed); rows = []
        for j, (a, b) in enumerate(P):
            u, v = (a, b) if g.random() < .5 else (b, a); rows.append((j, u, v, 1)); bad = nb[u] | {u}; got = set()
            while len(got) < k:
                w = int(g.integers(n))
                if w not in bad and w not in got: got.add(w); rows.append((j, u, w, 0))
        return pd.DataFrame(rows, columns=['pair_id', 'anchor', 'other', 'label'])
    deg = np.bincount(tr_g.ravel(), minlength=n).astype(float)
    return tr_g, deg, pairs(tr_g, K_TR, SEED_DATA + 1), pairs(va, K_EV, SEED_DATA + 2), pairs(te, K_EV, SEED_DATA + 3)

def fast_auc(df, s):
    S = s.reshape(-1, K_EV + 1); p = S[:, :1]; q = S[:, 1:]
    return float((((q < p).sum(1) + (q == p).sum(1) / 2) / K_EV).mean())

def run_m1(x, TR, TE):
    from sklearn.decomposition import TruncatedSVD; from sklearn.linear_model import LogisticRegression
    h = TruncatedSVD(128, random_state=0).fit_transform(x)
    Z = lambda D: np.hstack([h[D.anchor] * h[D.other], np.abs(h[D.anchor] - h[D.other])])
    Zt = Z(TR); mu, sd = Zt.mean(0), Zt.std(0); sd[sd == 0] = 1
    m = LogisticRegression(C=0.1, max_iter=3000).fit((Zt - mu) / sd, TR.label.values)
    return m.decision_function((Z(TE) - mu) / sd)

def run_gnn(model, x, edges, deg, TR, VA, TE, seed):
    from paper_models import SAGE, fit_kmeans; from dev_models import DFGTDev, dfgt_prepare_dev
    torch.manual_seed(seed); np.random.seed(seed)
    Y = dict(ids=list(range(len(x))), x=x, edges=[tuple(e) for e in edges] if model != 'nograph' else [], deg=deg if model != 'nograph' else np.zeros_like(deg))
    xt = torch.tensor(x)
    if model == 'sage': m = SAGE(x.shape[1], 128); P = SAGE.prepare(None, Y); lr, lam = 1e-2, 0.
    else:
        km = fit_kmeans({0: Y}, years=(0,)); P = dfgt_prepare_dev(Y, km)
        m = DFGTDev(x.shape[1], aug=(model == 'dfgt')); lr, lam = 3e-3, 0.1
    T = lambda D: (torch.tensor(D.anchor.values), torch.tensor(D.other.values))
    (ut, vt), (uv, vv), (ue, ve) = T(TR), T(VA), T(TE); y = torch.tensor(TR.label.values, dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=lr); best, bad, state = -1, 0, None
    for ep in range(150):
        m.train(); opt.zero_grad(); h, ssl = m.embed(xt, P)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m.dec(h, ut, vt), y) + lam * ssl; loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            h, _ = m.embed(xt, P); a = fast_auc(VA, torch.sigmoid(m.dec(h, uv, vv)).cpu().numpy())
            if a > best: best, bad, state = a, 0, torch.sigmoid(m.dec(h, ue, ve)).cpu().numpy()
            else:
                bad += 1
                if bad >= 15: break
    return state

def run(data, r, seeds):
    x, E = load(data); edges, deg, TR, VA, TE = split(len(x), E, r)
    out = TE.copy(); out['mindeg'] = np.minimum(deg[TE.anchor], deg[TE.other]); rt = []
    th20 = np.quantile(deg, .2); out['low20'] = (out.mindeg <= th20).astype(int)
    t0 = time.time(); out['M1'] = run_m1(x, TR, TE); rt.append(dict(data=data, r=r, model='M1', seconds=round(time.time() - t0, 2), gpu_mb=0))
    for model in ('sage', 'dfgt', 'nograph'):
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); per = [run_gnn(model, x, edges, deg, TR, VA, TE, sd) for sd in range(1, seeds + 1)]
        gpu = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0
        rt.append(dict(data=data, r=r, model=model, seconds=round((time.time() - t0) / seeds, 2), gpu_mb=round(gpu)))
        out[model] = np.mean(per, 0)
        for k, s in enumerate(per, 1): out[f'{model}_s{k}'] = s
        print(data, r, model, 'test AUC', round(fast_auc(TE, out[model].values), 4), f'{rt[-1]["seconds"]}s/seed', f'{rt[-1]["gpu_mb"]}MB', flush=True)
    out.to_csv(OUT / f'{data}_r{r:.1f}.csv', index=False)
    f = OUT / 'runtime.csv'; old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    old = old[~((old.get('data') == data) & (old.get('r') == r))] if len(old) else old
    pd.concat([old, pd.DataFrame(rt)]).to_csv(f, index=False)
    print(data, r, 'edges', len(E), 'train graph edges', len(edges), 'isolated test positives', int(((TE.label == 1) & (out.mindeg == 0)).sum()), 'of', int(TE.label.sum()), flush=True)

def evaluate():
    """每个数据集、每个 r：各模型组内 AUC（全部 / 低连接 / 高连接 / 孤立 / 最低 20%），以及 DFGT − 无图对照等差值的按锚聚类配对 bootstrap 区间
    和种子配对（孤立组差值 > 0 的种子数）；命题 2 的主判定与次要判定。"""
    res = {}; lines = []
    for data in DATA:
        for r in RS:
            f = OUT / f'{data}_r{r:.1f}.csv'
            if not f.exists(): continue
            S = pd.read_csv(f); S['pair_id'] = S.pair_id.astype(str); S['pos_group'] = ''
            models = [c for c in ('M1', 'sage', 'dfgt', 'nograph') if c in S]
            W = pd.concat([within_metrics(S.assign(score=S[c]), 'score')[['AUC']].rename(columns={'AUC': c}) for c in models], axis=1)
            pos = S[S.label == 1].set_index('pair_id'); W = W.join(pos[['anchor', 'mindeg', 'low20']])
            seedcols = [c for c in S if c.startswith('dfgt_s')]
            for k in range(1, len(seedcols) + 1):
                for mdl in ('dfgt', 'nograph'):
                    W[f'{mdl}_s{k}'] = within_metrics(S.assign(score=S[f'{mdl}_s{k}']), 'score').AUC
            grp = {'all': W.index == W.index, 'low': (W.mindeg <= 1).values, 'high': (W.mindeg >= 2).values, 'iso': (W.mindeg == 0).values, 'low20': (W.low20 == 1).values}
            ua = np.unique(W.anchor.values); ix = {a: np.where(W.anchor.values == a)[0] for a in ua}; rng = np.random.default_rng(20261103)
            draws = [np.concatenate([ix[a] for a in rng.choice(ua, len(ua))]) for _ in range(1000)]
            def ci(d, m):
                v = [np.nanmean(d[dr][m[dr]]) if m[dr].any() else np.nan for dr in draws]
                return [round(float(np.nanmean(d[m])), 4), round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)] if m.any() else [None] * 3
            R = dict(n={g: int(m.sum()) for g, m in grp.items()}, auc={c: {g: round(float(W.loc[m, c].mean()), 4) if m.any() else None for g, m in grp.items()} for c in models}, diff={})
            for a, b in (('dfgt', 'nograph'), ('sage', 'nograph'), ('dfgt', 'M1'), ('nograph', 'M1')):
                if a in W and b in W: R['diff'][f'{a} − {b}'] = {g: ci((W[a] - W[b]).values, m) for g, m in grp.items()}
            if seedcols: R['seed_pos_iso'] = int(sum(((W[f'dfgt_s{k}'] - W[f'nograph_s{k}'])[grp['iso']].mean() > 0) for k in range(1, len(seedcols) + 1))) if grp['iso'].any() else None
            res.setdefault(data, {})[f'{r:.1f}'] = R
            d = R['diff'].get('dfgt − nograph', {})
            lines.append(f"{data:9s} r={r:.1f} n(all/iso)={R['n']['all']}/{R['n']['iso']}  DFGT−nograph: all {d.get('all')}  iso {d.get('iso')}  low20 {d.get('low20')}  seeds>0(iso) {R.get('seed_pos_iso')}")
    # 命题 2 判定
    verdict = {}
    for data, rr in res.items():
        ok = []; allg = []
        for r, R in rr.items():
            d = R['diff'].get('dfgt − nograph', {})
            if float(r) > 0 and d.get('iso', [None])[0] is not None:
                e, lo, hi = d['iso']; ok.append((lo <= 0 <= hi) or abs(e) < .01)
            if d.get('all', [None])[0] is not None: allg.append((float(r), d['all'][0]))
        rho = pd.Series([g for _, g in allg]).corr(pd.Series([r for r, _ in allg]), method='spearman') if len(allg) >= 3 else None
        verdict[data] = dict(supported=bool(ok) and all(ok), n_r_checked=len(ok), spearman_all_gain_vs_r=None if rho is None else round(float(rho), 3))
    res['verdict'] = dict(per_data=verdict, n_supported=sum(v['supported'] for v in verdict.values()), n_data=len(verdict))
    json.dump(res, open(OUT / 'summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(OUT / 'summary.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n' + json.dumps(res['verdict'], ensure_ascii=False, indent=1))
    print('\n'.join(lines)); print(json.dumps(res['verdict'], ensure_ascii=False, indent=1))

# ======================= 第 18 节：C1（边丢弃）、C2（新基线）、C3（SEAL） =======================
C = OUT / 'c'; C.mkdir(exist_ok=True)
KR_CFG = L / 'out_dev_gpu10'   # 基线配置用韩国数据上选出的（第 18.2 节）

def drop_nodes(n, edges, deg, q, seed):
    """第 18.1 节：在度 > 0 的节点中抽比例 q（种子 20261106 + 100 + s），删除它们在消息传递图中的全部边。"""
    rng = np.random.default_rng(SEED_DATA + 100 + seed); cand = np.where(deg > 0)[0]
    d = np.zeros(n, bool); d[rng.choice(cand, int(round(q * len(cand))), replace=False)] = True
    e = edges[~(d[edges[:, 0]] | d[edges[:, 1]])]; return e, np.bincount(e.ravel(), minlength=n).astype(float)

def run_model(model, x, edges, deg, TR, VA, TE, seed):
    """model：sage / dfgt（可带 -ed10、-ed30 后缀）或 baselines.BASELINES 中的名字。返回（测试分数，验证分数，最佳验证 AUC）。"""
    import copy
    from paper_models import SAGE, fit_kmeans; from dev_models import DFGTDev, dfgt_prepare_dev; from baselines import BASELINES
    torch.manual_seed(seed); np.random.seed(seed); n = len(x); xt = torch.tensor(x)
    base, q = (model.split('-ed')[0], int(model.split('-ed')[1]) / 100) if '-ed' in model else (model, 0.)
    Yf = dict(ids=list(range(n)), x=x, edges=[tuple(e) for e in edges], deg=deg)
    if q > 0:
        e2, d2 = drop_nodes(n, edges, deg, q, seed); Yt = dict(Yf, edges=[tuple(e) for e in e2], deg=d2)
    else: Yt = Yf
    lam, fair, two = 0., False, False
    if base == 'sage':
        m = SAGE(x.shape[1], 128); prep = lambda Y: SAGE.prepare(None, Y); lr = 1e-2
    elif base == 'dfgt':
        km = fit_kmeans({0: Yf}, years=(0,)); m = DFGTDev(x.shape[1], aug=True); prep = lambda Y: dfgt_prepare_dev(Y, km); lr, lam = 3e-3, 0.1
    else:
        cfg = json.load(open(KR_CFG / f"base_config_{'coldbrew' if base == 'coldbrew2' else base}.json"))['cfg']
        m = BASELINES[base](x.shape[1], hid=cfg['hid']); prep = m.prepare; lr, lam, fair, two = cfg['lr'], 1., base == 'degfair', base == 'coldbrew2'
    Pt = prep(Yt); Pe = Pt if Yt is Yf else prep(Yf)
    T = lambda D: (torch.tensor(D.anchor.values), torch.tensor(D.other.values))
    (ut, vt), (uv, vv), (ue, ve) = T(TR), T(VA), T(TE); y = torch.tensor(TR.label.values, dtype=torch.float32)
    def fit(params, kd_only=False):
        opt = torch.optim.Adam(params, lr=lr); best, bad, sd, out = -1, 0, None, None
        for ep in range(150):
            m.train(); opt.zero_grad(); h, aux = m.embed(xt, Pt)
            if kd_only: loss = aux
            else:
                pl = torch.nn.functional.binary_cross_entropy_with_logits(m.dec(h, ut, vt), y, reduction='none'); loss = pl.mean() + lam * aux
                if fair: loss = loss + m.fair_loss(pl, Pt, ut, vt)
            loss.backward(); opt.step(); m.eval()
            with torch.no_grad():
                h, _ = m.embed(xt, Pe); sv = torch.sigmoid(m.dec(h, uv, vv)).cpu().numpy(); a = fast_auc(VA, sv)
                if a > best: best, bad, out, sd = a, 0, (torch.sigmoid(m.dec(h, ue, ve)).cpu().numpy(), sv), copy.deepcopy(m.state_dict())
                else:
                    bad += 1
                    if bad >= 15: break
        m.load_state_dict(sd); return out[0], out[1], best
    if two:
        m.stage = 1; fit(m.stage_params()); m.stage = 2; return fit(m.stage_params(), kd_only=True)
    return fit(list(m.parameters()))

def _upsert(f, row, keys=('data', 'r', 'model')):
    old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=list(keys))
    mask = np.ones(len(old), bool)
    for k in keys: mask &= (old[k] == row[k]).values
    pd.concat([old[~mask], pd.DataFrame([row])]).to_csv(f, index=False)

def run_c(data, r, seeds, models):
    """C1 / C2：每个模型 10 个种子，测试分数写 bench/out/c/<model>_<data>_r<r>.csv；种子平均后的验证 AUC 写 c/val_auc.csv；时间和显存写 c/runtime.csv。"""
    x, E = load(data); edges, deg, TR, VA, TE = split(len(x), E, r)
    for model in models:
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); err = ''
        try: res = [run_model(model, x, edges, deg, TR, VA, TE, sd) for sd in range(1, seeds + 1)]
        except torch.OutOfMemoryError:
            res = None; err = 'OOM'; torch.cuda.empty_cache()
        gpu = round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0; sec = round((time.time() - t0) / seeds, 2)
        _upsert(C / 'runtime.csv', dict(data=data, r=r, model=model, seconds=sec, gpu_mb=gpu, error=err))
        if res is None:
            print(data, r, model, 'OOM', flush=True); continue
        out = TE[['pair_id', 'anchor', 'other', 'label']].copy(); out[model] = np.mean([a for a, _, _ in res], 0)
        for k, (a, _, _) in enumerate(res, 1): out[f'{model}_s{k}'] = a
        out.to_csv(C / f'{model}_{data}_r{r:.1f}.csv', index=False)
        va = fast_auc(VA, np.mean([b for _, b, _ in res], 0)); _upsert(C / 'val_auc.csv', dict(data=data, r=r, model=model, val_auc=round(va, 5)))
        print(data, r, model, 'val', round(va, 4), 'test', round(fast_auc(TE, out[model].values), 4), f'{sec}s/seed {gpu}MB', flush=True)

SEAL_SEEDS = 3
def run_seal(data, r):
    """C3：SEAL 在公开基准上（第 18.3 节的规模控制），配置用韩国数据上选出的，3 个种子。"""
    import seal as S
    from sklearn.decomposition import TruncatedSVD
    x, E = load(data); edges, deg, TR, VA, TE = split(len(x), E, r); rng = np.random.default_rng(SEED_DATA)
    h = TruncatedSVD(128, random_state=0).fit_transform(x).astype(np.float32); h = (h - h.mean(0)) / h.std(0)
    TRp = TR.groupby('pair_id').head(1 + 5)   # 每个正例 + 前 5 个负例
    pid = TRp.pair_id.unique()
    if len(pid) > 20000: TRp = TRp[TRp.pair_id.isin(rng.choice(pid, 20000, replace=False))]
    def sub(D, k):
        p = D.pair_id.unique(); return D[D.pair_id.isin(rng.choice(p, k, replace=False))] if len(p) > k else D
    VAs, TEs = (sub(VA, 1000), sub(TE, 2000)) if data == 'Photo' else (VA, TE)
    t0 = time.time(); g = S.Graph(len(x), edges)
    st = {k: S.extract(g, D.anchor.values, D.other.values) for k, D in (('tr', TRp), ('va', VAs), ('te', TEs))}; tex = round(time.time() - t0, 1)
    print(data, r, 'SEAL subgraphs', {k: len(v['noff']) - 1 for k, v in st.items()}, f'{tex}s', flush=True)
    cfg = json.load(open(KR_CFG / 'base_config_seal.json'))['cfg']; xt = torch.tensor(h)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = [S.train(xt, st['tr'], TRp.label.values, st['va'], lambda s_: fast_auc(VAs, s_), st['te'], sd, cfg['hid'], cfg['lr']) for sd in range(1, SEAL_SEEDS + 1)]
    gpu = round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0; sec = round((time.time() - t0) / SEAL_SEEDS, 2)
    out = TEs[['pair_id', 'anchor', 'other', 'label']].copy(); out['seal'] = np.mean([c for _, _, c in res], 0)
    for k, (_, _, c) in enumerate(res, 1): out[f'seal_s{k}'] = c
    out.to_csv(C / f'seal_{data}_r{r:.1f}.csv', index=False)
    _upsert(C / 'runtime.csv', dict(data=data, r=r, model='seal', seconds=sec, gpu_mb=gpu, error='', extract_s=tex))
    print(data, r, 'seal test', round(fast_auc(TEs, out.seal.values), 4), f'{sec}s/seed {gpu}MB', flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--data'); ap.add_argument('--r', type=float); ap.add_argument('--seeds', type=int, default=10); ap.add_argument('--models', default='')
    ap.add_argument('--device', default='cuda'); A = ap.parse_args()
    if A.device != 'cpu' and torch.cuda.is_available(): torch.set_default_device(A.device)
    if A.cmd == 'run': run(A.data, A.r, A.seeds)
    elif A.cmd == 'eval': evaluate()
    elif A.cmd == 'run-c': run_c(A.data, A.r, A.seeds, A.models.split(','))
    elif A.cmd == 'run-seal': run_seal(A.data, A.r)
