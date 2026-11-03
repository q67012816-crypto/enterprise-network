"""度偏差论文研究（protocol.md A7）的运行脚本。输出目录 out_paper/。
  python src/paper_run.py baselines                 # M1、M1+scores、四个启发式：只用验证年做选择
  python src/paper_run.py search --model sage --ci 0 # GNN 网格 × 5 种子：训练年训练、验证年早停与选择
  python src/paper_run.py test                      # 用冻结的选择给测试年打分；只能运行一次
  python src/paper_run.py eval                      # 只读冻结分数：组内指标、度偏差差距、bootstrap、成功标准 S_A–S_D
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import L, YEARS, load_pairs, node_tables, within_metrics, boot_ci
from paper_models import SAGE, DFGT, dfgt_prepare, fit_kmeans

O = L / 'out_paper'; O.mkdir(exist_ok=True); (O / 'models').mkdir(exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--model', default='sage'); ap.add_argument('--ci', type=int, default=0)
ap.add_argument('--threads', type=int, default=4); ARGS = ap.parse_args(); torch.set_num_threads(ARGS.threads)
X = load_pairs(); SPLIT_YEAR = {'train': (2021, 2022), 'validation': (2023,), 'test': (2024,)}
GRID = {'sage': list(itertools.product([64, 128], [1e-3, 1e-2])), 'dfgt': list(itertools.product([1e-3, 3e-3], [0.1, 1.0]))}
SEEDS = [1, 2, 3, 4, 5]
fam = lambda m, ci: f'{m}{"+ci" if ci else ""}'

def fast_within_auc(df, s):
    """训练过程中用的快速组内 AUC（每个正例后面紧跟它的 100 个负例时直接 reshape；否则退回逐组计算）。"""
    y = df.label.values
    if len(y) % 101 == 0 and (y.reshape(-1, 101)[:, 0] == 1).all() and (y.reshape(-1, 101)[:, 1:] == 0).all():
        S = s.reshape(-1, 101); p = S[:, :1]; n = S[:, 1:]
        return float((((n < p).sum(1) + (n == p).sum(1) / 2) / 100).mean())
    d = df.assign(_s=s); return float(within_metrics(d, '_s').AUC.mean())

# ---------------- 基线 ----------------
def heur(Y):
    """在 G_t 上为每个点对计算 CN、AA、Jaccard、PA（只用当年的持股图，不含共同投资）。"""
    nb = {i: set() for i in range(len(Y['ids']))}
    for i, j in Y['edges']: nb[i].add(j); nb[j].add(i)
    def f(a, o):
        A_, B_ = nb[a], nb[o]; c = A_ & B_
        return len(c), sum(1 / np.log(len(nb[z])) for z in c if len(nb[z]) > 1), len(c) / len(A_ | B_) if A_ | B_ else 0., len(A_) * len(B_)
    return f

def design(Ys, df):
    """点对的设计矩阵：双方节点特征（同 M1 口径）、同行业、同市场；另返回四个启发式分数。"""
    rows, H = [], []
    for t, g in df.groupby('fy'):
        Y = Ys[t]; a = np.array([Y['idx'][c] for c in g.anchor]); o = np.array([Y['idx'][c] for c in g.other]); F = Y['F']
        ind = F[:, 20:41].argmax(1); mk = F[:, 41:43].argmax(1) + 10 * (F[:, 41:43].sum(1) == 0)
        Z = np.hstack([Y['x'][a], Y['x'][o], (ind[a] == ind[o])[:, None], (mk[a] == mk[o])[:, None]])
        h = heur(Y); hh = np.array([h(i, j) for i, j in zip(a, o)])
        rows.append(pd.DataFrame(Z, index=g.index)); H.append(pd.DataFrame(hh, index=g.index, columns=['CN', 'AA', 'Jaccard', 'PA']))
    return pd.concat(rows).loc[df.index], pd.concat(H).loc[df.index]

def baselines(score_test=False):
    """拟合 M1、M1+scores（只用训练年），用验证组内 AUC 选 C 和最强启发式；score_test=True 时用冻结选择给测试年打分。"""
    Ys = node_tables(False); Z, H = design(Ys, X)
    Z2 = pd.concat([Z, np.log1p(H.CN), H.AA, H.Jaccard, np.log1p(H.PA)], axis=1)
    Z.columns = [f'f{i}' for i in range(Z.shape[1])]; Z2.columns = [f'f{i}' for i in range(Z2.shape[1])]
    tr, va, te = ((X.split == s).values for s in ('train', 'validation', 'test'))
    def fit(M_, C):
        mu, sd = M_[tr].mean(), M_[tr].std().replace(0, 1)
        m = LogisticRegression(C=C, max_iter=5000).fit((M_[tr] - mu) / sd, X.label[tr]); return lambda msk: m.predict_proba((M_[msk] - mu) / sd)[:, 1]
    self = O / 'selection_baselines.json'
    if not score_test:
        sel = {}; V = X[va].copy()
        for name, M_ in (('M1', Z), ('M1+scores', Z2)):
            r = {C: fast_within_auc(V, fit(M_, C)(va)) for C in (0.01, 0.1, 1, 10)}; sel[name] = dict(C=max(r, key=r.get), val_within_AUC=r)
            V[name] = fit(M_, sel[name]['C'])(va)
        hv = {h: fast_within_auc(V, H[h].values[va]) for h in ('CN', 'AA', 'Jaccard', 'PA')}; sel['best_heuristic'] = max(hv, key=hv.get); sel['heuristic_val_within_AUC'] = hv
        for h in hv: V[h] = H[h].values[va]
        V.to_csv(O / 'val_scores_baselines.csv', index=False); json.dump(sel, open(self, 'w'), indent=1); print(json.dumps(sel, indent=1)); return
    sel = json.load(open(self)); T = X[te][['pair_id', 'fy', 'anchor', 'other', 'label', 'pos_group']].copy()
    T['M1'] = fit(Z, sel['M1']['C'])(te); T['M1+scores'] = fit(Z2, sel['M1+scores']['C'])(te)
    for h in ('CN', 'AA', 'Jaccard', 'PA'): T[h] = H[h].values[te]
    return T

# ---------------- GNN ----------------
def prep(model, ci):
    """读取节点表并为每年准备图结构输入；DFGT 还需要 k-means 社区。"""
    Ys = node_tables(bool(ci)); km = fit_kmeans(Ys) if model == 'dfgt' else None
    Ps = {t: (dfgt_prepare(Ys[t], km) if model == 'dfgt' else SAGE.prepare(None, Ys[t])) for t in YEARS}
    xs = {t: torch.tensor(Ys[t]['x']) for t in YEARS}
    return Ys, Ps, xs

def pair_idx(Ys, df):
    """按年把点对 (anchor, other) 换成节点下标。"""
    return {t: (torch.tensor([Ys[t]['idx'][a] for a in g.anchor]), torch.tensor([Ys[t]['idx'][o] for o in g.other]), g.index.values) for t, g in df.groupby('fy')}

def make(model, fin, cfg):
    return SAGE(fin, cfg[0]) if model == 'sage' else DFGT(fin)

def score(m, Ps, xs, PI, n):
    """评估模式下给一组点对打分（sigmoid 概率）。"""
    m.eval(); out = np.zeros(n)
    with torch.no_grad():
        for t, (u, v, ix) in PI.items(): h, _ = m.embed(xs[t], Ps[t]); out[ix] = torch.sigmoid(m.dec(h, u, v)).numpy()
    return out

def train_one(model, cfg, seed, Ys, Ps, xs, PI_tr, PI_va, VA):
    """训练一个模型：训练年（2021→22、2022→23）的正负例 + 自监督损失（DFGT），按验证组内 AUC 早停（patience 15，最多 150 轮）。"""
    torch.manual_seed(seed); np.random.seed(seed)
    fin = xs[2021].shape[1]; m = make(model, fin, cfg); lr = cfg[1] if model == 'sage' else cfg[0]; lam = 0. if model == 'sage' else cfg[1]
    opt = torch.optim.Adam(m.parameters(), lr=lr); y = torch.tensor(X.label.values, dtype=torch.float32)
    best, state, bad, ep_best = -1, None, 0, 0; ntr = sum(len(v[2]) for v in PI_tr.values())
    for ep in range(1, 151):
        m.train(); opt.zero_grad(); loss = 0.
        for t, (u, v, ix) in PI_tr.items():
            h, ssl = m.embed(xs[t], Ps[t])
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(m.dec(h, u, v), y[ix], reduction='sum') / ntr + lam * ssl / len(PI_tr)
        loss.backward(); opt.step()
        a = fast_within_auc(VA, score(m, Ps, xs, PI_va, len(X))[VA.index])
        if a > best: best, bad, ep_best, state = a, 0, ep, {k: w.clone() for k, w in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 15: break
    m.load_state_dict(state)
    return m, dict(val_within_AUC=best, best_epoch=ep_best, epochs_run=ep)

def search(model, ci):
    """网格 × 5 种子；按 5 种子平均验证组内 AUC 选配置；保存每个模型和验证分数。不读取测试标签。"""
    f = fam(model, ci); Ys, Ps, xs = prep(model, ci)
    TR, VA = X[X.split == 'train'], X[X.split == 'validation']; PI_tr, PI_va = pair_idx(Ys, TR), pair_idx(Ys, VA)
    rows = []; t0 = time.time(); VS = VA[['pair_id', 'label']].copy()
    for cfg in GRID[model]:
        for sd in SEEDS:
            m, r = train_one(model, cfg, sd, Ys, Ps, xs, PI_tr, PI_va, VA)
            name = f'{f}_{cfg[0]}_{cfg[1]}_s{sd}'; torch.save(m.state_dict(), O / 'models' / f'{name}.pt')
            VS[name] = score(m, Ps, xs, PI_va, len(X))[VA.index]
            rows.append(dict(family=f, config=f'{cfg[0]}_{cfg[1]}', seed=sd, **r)); print(rows[-1], f'{time.time() - t0:.0f}s', flush=True)
    R = pd.DataFrame(rows); R.to_csv(O / f'val_runs_{f}.csv', index=False); VS.to_csv(O / f'val_scores_{f}.csv', index=False)
    S = R.groupby('config').val_within_AUC.mean().sort_values(ascending=False)
    json.dump(dict(family=f, selected=S.index[0], val_mean_within_AUC=S.to_dict()), open(O / f'selection_{f}.json', 'w'), indent=1)
    print('selected', f, S.index[0])

def test():
    """只运行一次：用冻结的选择给测试年（2024→25）打分，GNN 取 5 个种子预测的平均。输出已存在时拒绝运行。"""
    outf = O / 'test_scores.csv'
    if outf.exists(): sys.exit('测试已打分过一次，拒绝重复运行')
    fams = [fam(m, c) for m in ('sage', 'dfgt') for c in (0, 1)]
    for f in fams + ['baselines']:
        if not (O / f'selection_{f}.json').exists(): sys.exit(f'缺少冻结的选择：selection_{f}.json')
    T = baselines(score_test=True); TE = X[X.split == 'test']
    for model in ('sage', 'dfgt'):
        for ci in (0, 1):
            f = fam(model, ci); cfg_s = json.load(open(O / f'selection_{f}.json'))['selected']; a, b = cfg_s.split('_')
            cfg = (int(a), float(b)) if model == 'sage' else (float(a), float(b))
            Ys, Ps, xs = prep(model, ci); PI = pair_idx(Ys, TE); S = []
            for sd in SEEDS:
                m = make(model, xs[2021].shape[1], cfg); m.load_state_dict(torch.load(O / 'models' / f'{f}_{a}_{b}_s{sd}.pt')); S.append(score(m, Ps, xs, PI, len(X))[TE.index])
            T[f] = np.mean(S, 0)
    T.to_csv(outf, index=False)
    json.dump(dict(test_scored_once=True, time=time.strftime('%Y-%m-%d %H:%M:%S %z')), open(O / 'TEST_DONE.json', 'w'))
    print('测试集打分完成（仅此一次）')

def evaluate():
    """只读冻结的验证/测试分数：组内指标、按正例所在组的度偏差差距 Δ、bootstrap 区间，以及成功标准 S_A–S_D。"""
    sel_b = json.load(open(O / 'selection_baselines.json')); bh = sel_b['best_heuristic']
    fams = [fam(m, c) for m in ('sage', 'dfgt') for c in (0, 1)]
    val = {f: json.load(open(O / f'selection_{f}.json')) for f in fams}
    vbest = {f: max(v['val_mean_within_AUC'].values()) for f, v in val.items()}
    T = pd.read_csv(O / 'test_scores.csv', dtype={'anchor': str, 'other': str, 'pair_id': str})
    models = ['M1', 'M1+scores', bh] + fams
    W = None
    for m in models:
        w = within_metrics(T, m).add_prefix(f'{m}|').rename(columns={f'{m}|anchor': 'anchor', f'{m}|pos_group': 'pos_group'})
        W = w if W is None else W.join(w[[c for c in w.columns if c.startswith(f'{m}|')]])
    W = W.reset_index(); res = dict(best_heuristic=bh, val_best_within_AUC=vbest, n_pos=W.pos_group.value_counts().to_dict(), cells={})
    for cell, msk in [('all', np.ones(len(W), bool))] + [(g, (W.pos_group == g).values) for g in ('both_connected', 'one_isolated', 'both_isolated')]:
        U = W[msk]; res['cells'][cell] = {f'{m}|{k}': float(U[f'{m}|{k}'].mean()) for m in models for k in ('AUC', 'RR', 'H10')}
        TP = T[T.pair_id.isin(U.pair_id)]
        res['cells'][cell].update({f'{m}|AP': float(average_precision_score(TP.label, TP[m])) for m in models})
    gap = lambda m: (lambda U: U.loc[U.pos_group == 'both_connected', f'{m}|AUC'].mean() - U.loc[U.pos_group == 'one_isolated', f'{m}|AUC'].mean())
    allauc = lambda m: (lambda U: U[f'{m}|AUC'].mean())
    best_graph = max(fams, key=vbest.get); best_fam_noci = 'sage' if vbest['sage'] >= vbest['dfgt'] else 'dfgt'
    res['gaps'] = {m: boot_ci(W, gap(m)) for m in models}
    res['S_A_gap_SAGE'] = res['gaps']['sage']
    res['S_B_gap_reduction_SAGE_minus_DFGT'] = boot_ci(W, lambda U: gap('sage')(U) - gap('dfgt')(U))
    res['S_B_overall_DFGT_minus_SAGE'] = boot_ci(W, lambda U: allauc('dfgt')(U) - allauc('sage')(U))
    res['S_C_family'] = best_fam_noci; res['S_C_ci_minus_noci'] = boot_ci(W, lambda U: allauc(best_fam_noci + '+ci')(U) - allauc(best_fam_noci)(U))
    res['S_D_best_graph'] = best_graph; res['S_D_best_graph_minus_M1'] = boot_ci(W, lambda U: allauc(best_graph)(U) - allauc('M1')(U))
    res['diff_vs_M1_by_group'] = {g: {m: boot_ci(W[W.pos_group == g], lambda U, m=m: allauc(m)(U) - allauc('M1')(U)) for m in fams + ['M1+scores']} for g in ('both_connected', 'one_isolated')}
    ok = lambda r: r[1] > 0
    res['verdict'] = dict(S_A=ok(res['S_A_gap_SAGE']), S_B=ok(res['S_B_gap_reduction_SAGE_minus_DFGT']) and res['S_B_overall_DFGT_minus_SAGE'][0] > -0.02,
                          S_C=ok(res['S_C_ci_minus_noci']), S_D=ok(res['S_D_best_graph_minus_M1']))
    W.to_csv(O / 'test_within.csv', index=False); json.dump(res, open(O / 'test_eval.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))

if ARGS.cmd == 'baselines': baselines()
elif ARGS.cmd == 'search': search(ARGS.model, ARGS.ci)
elif ARGS.cmd == 'test': test()
elif ARGS.cmd == 'eval': evaluate()
