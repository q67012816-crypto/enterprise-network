"""开发阶段实验（只用训练年和验证年；不读测试年 2024→25）。数据：data_v3x（2019→20 起）。
滚动三折：F1 训练 2019–2020 → 验证 2021；F2 训练 2019–2021 → 验证 2022；F3 训练 2019–2022 → 验证 2023。
低连接组 = 新边两端中较小的度 ≤ 1（与 DegFairGT"度数最低一组"的思路一致）；高连接组 = 两端度都 ≥ 2。
用法：python src/paper_dev.py run --variant dfgt      # 单个变体，3 折 × 3 种子，输出 out_dev/scores_<variant>.csv
      python src/paper_dev.py eval                     # 汇总所有变体：整体/低/高连接组组内 AUC、差距、配对 bootstrap
超参数沿用 A7 在验证集上选出的值（SAGE：hidden 128、lr 1e-2；DFGT：lr 3e-3、λ 0.1；M1 C=0.1，M1+scores C=10），本阶段不再调参。"""
import argparse, collections, json, sys, time
RT = []   # 新基线的训练时间与显存记录
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import FI, rd, within_metrics
try:   # 逻辑回归变体不需要 torch
    import torch
    from paper_models import SAGE, fit_kmeans
    from dev_models import DFGTDev, dfgt_prepare_dev
    from dual_models import DualChannel, dcm_prepare
    from baselines import BASELINES, BASE_GRID
except ImportError: torch = None

L = Path(__file__).resolve().parents[1]; P0 = L.parent; D = L / 'data_v3x'; FE = O = L / 'out_dev'   # FE：点对特征文件所在目录；O：分数输出目录（可用 --out 改）
FOLDS = {'F1': ((2019, 2020), 2021), 'F2': ((2019, 2020, 2021), 2022), 'F3': ((2019, 2020, 2021, 2022), 2023)}
SEEDS = [1, 2, 3]
VARIANTS = {   # 名称: (模型, 选项)
    'M1': ('lr', dict(C=0.1, target=False, scores=False)), 'M1+scores': ('lr', dict(C=10, target=False, scores=True)),
    'M1+T': ('lr', dict(C=0.1, target=True, scores=False)),
    'sage': ('sage', dict(target=False)), 'sage+T': ('sage', dict(target=True)),
    'dfgt': ('dfgt', dict(target=False)), 'dfgt-noaug': ('dfgt', dict(target=False, aug=False)),
    'dfgt-nosim': ('dfgt', dict(target=False, use_sim=False)), 'dfgt-nossl': ('dfgt', dict(target=False, lam=0.)),
    'dfgt0': ('dfgt', dict(target=False, zero_k=5)),
    'dfgt-soft': ('dfgt', dict(target=False, soft_eval=True)), 'dfgt0-soft': ('dfgt', dict(target=False, zero_k=5, soft_eval=True)),
    'dfgt-soft-nossl': ('dfgt', dict(target=False, soft_eval=True, lam=0.)),
    'dfgt-nograph': ('dfgt', dict(target=False, aug=False, nograph=True)),   # 对照：同一架构但去掉所有边（只剩自环），检验提升是否来自图 'dfgt+T': ('dfgt', dict(target=True)), 'dfgt0+T': ('dfgt', dict(target=True, zero_k=5)),
    # 第 1、2 项（pair_feats.py）：port = 投资组合匹配，ind = 细行业；点对特征在 GNN 里以线性项加到解码器 logit 上
    'M1+ind': ('lr', dict(C=0.1, target=False, scores=False, pf=('ind',))), 'M1+port': ('lr', dict(C=0.1, target=False, scores=False, pf=('port',))),
    'M1+fine': ('lr', dict(C=0.1, target=False, scores=False, pf=('fine',))), 'M1+hist': ('lr', dict(C=0.1, target=False, scores=False, pf=('hist',))),
    'M1+port+fine': ('lr', dict(C=0.1, target=False, scores=False, pf=('port', 'fine'))),
    # 第 3 项（interlock_feats.py）：董事兼任
    'M1+il': ('lr', dict(C=0.1, target=False, scores=False, pf=('il',))), 'M1+port+il': ('lr', dict(C=0.1, target=False, scores=False, pf=('port', 'il'))),
    'M1+pf+il': ('lr', dict(C=0.1, target=False, scores=False, pf=('port', 'ind', 'il'))),
    'dfgt+il': ('dfgt', dict(target=False, pf=('il',))), 'dfgt+pf+il': ('dfgt', dict(target=False, pf=('port', 'ind', 'il'))),
    'dfgt-nograph+il': ('dfgt', dict(target=False, aug=False, nograph=True, pf=('il',))),
    # 兼任关系作为图的边（与持股图合并做消息传递）：ile = 全部直接兼任边，ile_in = 只用含执行人员（非사외이사、非감사）的兼任边
    'dfgt+ile': ('dfgt', dict(target=False, ile='all')), 'dfgt+ile+il': ('dfgt', dict(target=False, ile='all', pf=('il',))),
    'dfgt+ile+pf+il': ('dfgt', dict(target=False, ile='all', pf=('port', 'ind', 'il'))), 'dfgt+ile_in': ('dfgt', dict(target=False, ile='inside')),
    'sage+ile': ('sage', dict(target=False, ile='all')),
    # 后融合（不调参）：两个已有变体的分数在每个验证年内转成百分位秩后等权平均
    'ens:dfgt|M1+il': ('ens', dict(parts=('dfgt', 'M1+il'))), 'ens:dfgt|M1': ('ens', dict(parts=('dfgt', 'M1'))),
    # 不含结构信息的版本：port + 细行业（same3/4/5），去掉 hist_has/aff*/rev*（它们依赖过去的持股关系）
    'dfgt+port+fine': ('dfgt', dict(target=False, pf=('port', 'fine'))), 'dfgt+port+fine+il': ('dfgt', dict(target=False, pf=('port', 'fine', 'il'))),
    'M1+port+fine+il': ('lr', dict(C=0.1, target=False, scores=False, pf=('port', 'fine', 'il'))),
    # 第 4 项（hyslr_feats.py）：最大股东及特殊关系人
    'M1+hs': ('lr', dict(C=0.1, target=False, scores=False, pf=('hs',))), 'M1+il+hs': ('lr', dict(C=0.1, target=False, scores=False, pf=('il', 'hs'))),
    'M1+pf+il+hs': ('lr', dict(C=0.1, target=False, scores=False, pf=('port', 'ind', 'il', 'hs'))),
    'M1+hs-lenient': ('lr', dict(C=0.1, target=False, scores=False, pf=('hs_len',))),
    'dfgt+pf+hs': ('dfgt', dict(target=False, pf=('port', 'ind', 'hs'))), 'dfgt+pf+il+hs': ('dfgt', dict(target=False, pf=('port', 'ind', 'il', 'hs'))),
    'dfgt+ile+pf+il+hs': ('dfgt', dict(target=False, ile='all', pf=('port', 'ind', 'il', 'hs'))),
    'dfgt+port+fine+il+hs': ('dfgt', dict(target=False, pf=('port', 'fine', 'il', 'hs'))), 'M1+port+fine+il+hs': ('lr', dict(C=0.1, target=False, scores=False, pf=('port', 'fine', 'il', 'hs'))),
    # 探索性（冻结前追加）：履历中的前任职公司（career_feats.py）与 5% 大量持股人（majorstock_feats.py）
    'M1+car': ('lr', dict(C=0.1, target=False, scores=False, pf=('car',))), 'M1+il+car': ('lr', dict(C=0.1, target=False, scores=False, pf=('il', 'car'))),
    'M1+ms': ('lr', dict(C=0.1, target=False, scores=False, pf=('ms',))), 'M1+il+ms': ('lr', dict(C=0.1, target=False, scores=False, pf=('il', 'ms'))),
    'dfgt+port+fine+il+car': ('dfgt', dict(target=False, pf=('port', 'fine', 'il', 'car'))), 'dfgt+port+fine+il+ms': ('dfgt', dict(target=False, pf=('port', 'fine', 'il', 'ms'))),
    # DCM（DEV_RESULTS 第 14 节）：A 关系分开聚合，B 度数门控点对证据，C 以 M1 + il 的 logit 为基础分数的残差学习；超参数见 dcm_config.json（tune 命令选出）
    'dcm': ('dcm', dict(rel=True, gate=True, resid=True)), 'dcm-noA': ('dcm', dict(rel=False, gate=True, resid=True)),
    'dcm-noB': ('dcm', dict(rel=True, gate=False, resid=True)), 'dcm-noC': ('dcm', dict(rel=True, gate=True, resid=False)),
    # TNSE 实验 B 的新基线（baselines.py，按论文重写，不是官方实现）；超参数见 base_config_<名>.json（tune-base 命令选出）
    'gae': ('base', dict(name='gae')), 'coldbrew': ('base', dict(name='coldbrew')), 'coldbrew2': ('base', dict(name='coldbrew2', cfg_from='coldbrew')),
    'seal': ('base', dict(name='seal')),   # 第 18.3 节（seal.py，不是官方实现）
    # 第 18.1 节：训练时的节点级边丢弃（每个训练年 G_t 中比例 q 的度 > 0 节点不参与消息传递，训练正例不变；验证年用完整 G_t）
    'dfgt-ed10': ('dfgt', dict(target=False, ed=0.1)), 'dfgt-ed30': ('dfgt', dict(target=False, ed=0.3)),
    'sage-ed10': ('sage', dict(target=False, ed=0.1)), 'sage-ed30': ('sage', dict(target=False, ed=0.3)), 'degfair': ('base', dict(name='degfair')), 'tailgnn': ('base', dict(name='tailgnn')),
    'M1+il-lenient': ('lr', dict(C=0.1, target=False, scores=False, pf=('il_len',))),   # 敏感性：保留 C_t 后才提交的更正年报
    'M1+pf': ('lr', dict(C=0.1, target=False, scores=False, pf=('port', 'ind'))),
    'dfgt+pf': ('dfgt', dict(target=False, pf=('port', 'ind'))), 'dfgt-nograph+pf': ('dfgt', dict(target=False, aug=False, nograph=True, pf=('port', 'ind'))),
}

def pf_matrix(X, groups, tr_years):
    """读取 out_dev/pair_feats.csv 中所选组的点对特征，与 X 行对齐；缺失填 0 并加"组合特征缺失"指示；
    用本折训练年的均值和标准差标准化。未计算的行（fy 2024）全为 0。输出 float32 数组。"""
    from pair_feats import PORT, IND
    from interlock_feats import IL
    MS = ['ms_common', 'ms_common_inst', 'ms_common_indiv', 'ms_has_a', 'ms_has_o']   # 与 majorstock_feats.MS 相同
    HS = ['hs_same_max', 'hs_shared', 'hs_corp', 'hs_exec_ao', 'hs_exec_oa', 'hs_has_a', 'hs_has_o']   # 与 hyslr_feats.HS 相同
    F = pd.read_csv(FE / 'pair_feats.csv', index_col='row').reindex(X.index)
    if 'il' in groups: F = F.join(pd.read_csv(FE / 'interlock_feats.csv', index_col='row').reindex(X.index))
    if 'hs' in groups: F = F.join(pd.read_csv(FE / 'hyslr_feats.csv', index_col='row').reindex(X.index))
    if 'hs_len' in groups: F = F.join(pd.read_csv(FE / 'hyslr_feats_lenient.csv', index_col='row').reindex(X.index))
    if 'car' in groups: F = F.join(pd.read_csv(FE / 'career_feats.csv', index_col='row').reindex(X.index))
    if 'ms' in groups: F = F.join(pd.read_csv(FE / 'majorstock_feats.csv', index_col='row').reindex(X.index))
    if 'il_len' in groups: F = F.join(pd.read_csv(FE / 'interlock_feats_lenient.csv', index_col='row').reindex(X.index))
    cols = [c for g in groups for c in {'port': PORT, 'ind': IND, 'fine': IND[:3], 'hist': IND[3:], 'il': IL, 'il_len': IL, 'hs': HS, 'hs_len': HS, 'car': ['car_ao', 'car_oa'], 'ms': MS}[g]]; Z = F[cols].copy()
    if 'port' in groups: Z['pf_miss'] = Z.pf_size_gap.isna().astype(float)
    Z = Z.fillna(0.); tr = X.fy.isin(tr_years).values; mu, sd = Z[tr].mean(), Z[tr].std().replace(0, 1)
    Z = (Z - mu) / sd; Z.loc[X.fy.ge(2024).values] = 0.
    return Z.values.astype(np.float32)

def sfile(v):
    """变体的分数文件路径。变体名里的 ':' '|' 在 Windows 文件名中非法（':' 会写进 NTFS 备用数据流），替换掉。"""
    return O / ('scores_' + v.replace(':', '_').replace('|', '-') + '.csv')

def load():
    """读取 data_v3x 的点对、节点、各年持股图（有向）和合格名单。"""
    X = pd.read_csv(D / 'pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str})
    N = pd.read_csv(D / 'nodes.csv', dtype={'corp_code': str, 'industry': str, 'market': str, 'fin_disc': str}, keep_default_na=False, na_values=[''])
    G = {t: rd(D / f'graph_{t}.csv') for t in range(2019, 2025)}
    return X, N, G

def target_features(G, t, ids):
    """被投资方一侧的"吸引力"特征（只用 C_t 前可见的信息）：
    上市年限（首次在 C 前提交年报的财年到 t 的年数，log1p）、是否新进入（≤1 年）、
    t 年及以前是否曾被任何上市公司持有（只看各年 as-of 持股图 G_s，s ≤ t，不用历史集合 H）、G_t 中的入度和出度（log1p）。"""
    A = rd(P0 / 'data/interim/filings_A001.csv')
    ann = A[A.report_nm.str.match(r'^사업보고서\s*\(') & ~A.report_nm.str.contains('연장신고')].copy()
    ann['fy'] = pd.to_numeric(ann.report_nm.str.extract(r'사업보고서\s*\((\d{4})\.')[0], errors='coerce')
    ann = ann[ann.rcept_dt.le(f'{t + 1}0430') & ann.fy.le(t)]
    first = ann.groupby('corp_code').fy.min()
    held = set().union(*[set(G[s].dst) for s in range(2019, t + 1)])
    indeg = collections.Counter(G[t].dst); outdeg = collections.Counter(G[t].src)
    age = np.array([t - first.get(c, t) for c in ids], dtype=float)
    return np.vstack([np.log1p(age), (age <= 1).astype(float), np.array([c in held for c in ids], float),
                      np.log1p([indeg[c] for c in ids]), np.log1p([outdeg[c] for c in ids])]).T

def node_tables(N, G, years, train_years, target):
    """各年节点表：节点 = 当年合格企业 ∪ G_t 中的企业；特征同 M1（截断、中位数填补、缺失指示、行业/市场 one-hot），
    预处理统计量只用本折的训练年；target=True 时追加被投资方特征。输出 dict 年份 → {ids, idx, x, edges, deg, F}。"""
    tr = N[N.fy.isin(train_years)]
    lo, hi = tr[FI].quantile(.005), tr[FI].quantile(.995); med = tr[FI].clip(lo, hi, axis=1).median()
    topind = tr.industry.value_counts().index[:20].tolist(); out = {}
    for t in years:
        Nt = N[N.fy == t].set_index('corp_code'); ids = sorted(set(Nt.index) | set(G[t].src) | set(G[t].dst))
        M = Nt.reindex(ids); cols = []
        for c in FI: cols += [M[c].isna().astype(float).values, M[c].clip(lo[c], hi[c]).fillna(med[c]).values]
        ind = M.industry.fillna('').values; mk = M.market.fillna('').values
        cols += [(ind == i).astype(float) for i in topind] + [(~np.isin(ind, topind)).astype(float)] + [(mk == m).astype(float) for m in ('Y', 'K')]
        Fm = np.vstack(cols).T
        if target: Fm = np.hstack([Fm, target_features(G, t, ids)])
        idx = {c: i for i, c in enumerate(ids)}
        e = sorted({(min(idx[a], idx[b]), max(idx[a], idx[b])) for a, b in zip(G[t].src, G[t].dst) if a != b})
        deg = np.zeros(len(ids))
        for i, j in e: deg[i] += 1; deg[j] += 1
        out[t] = dict(ids=ids, idx=idx, F=Fm, edges=e, deg=deg)
    trF = np.vstack([out[t]['F'] for t in train_years]); mu, sd = trF.mean(0), trF.std(0); sd[sd == 0] = 1
    for t in years: out[t]['x'] = ((out[t]['F'] - mu) / sd).astype(np.float32)
    return out

def pair_idx(Ys, df):
    return {t: (torch.tensor([Ys[t]['idx'][a] for a in g.anchor]), torch.tensor([Ys[t]['idx'][o] for o in g.other]), g.index.values) for t, g in df.groupby('fy')}

def fast_auc(df, s):
    """快速组内 AUC（每个正例后紧跟它的 100 个负例）。"""
    S = s.reshape(-1, 101); p = S[:, :1]; n = S[:, 1:]
    return float((((n < p).sum(1) + (n == p).sum(1) / 2) / 100).mean())

def heur_scores(Y, a, o):
    nb = collections.defaultdict(set)
    for i, j in Y['edges']: nb[i].add(j); nb[j].add(i)
    out = []
    for i, j in zip(a, o):
        c = nb[i] & nb[j]; u = nb[i] | nb[j]
        out.append((np.log1p(len(c)), sum(1 / np.log(len(nb[z])) for z in c if len(nb[z]) > 1), len(c) / len(u) if u else 0., np.log1p(len(nb[i]) * len(nb[j]))))
    return np.array(out)

def run_lr(X, Ys, tr_years, va_year, opt, all_logits=False):
    """逻辑回归（M1 及其变体）：双方节点特征 + 同行业 + 同市场（+ 可选启发式分数），只用本折训练年拟合。
    all_logits=True 时返回本折训练年和验证年所有点对的 logit（Series，索引同 X），供 DCM 作基础分数。"""
    def design(df):
        rows = []
        for t, g in df.groupby('fy'):
            Y = Ys[t]; a = np.array([Y['idx'][c] for c in g.anchor]); o = np.array([Y['idx'][c] for c in g.other]); F = Y['F']
            ind = F[:, 20:41].argmax(1); mk = F[:, 41:43].argmax(1) + 10 * (F[:, 41:43].sum(1) == 0)
            Z = [Y['x'][a], Y['x'][o], (ind[a] == ind[o])[:, None], (mk[a] == mk[o])[:, None]]
            if opt['scores']: Z.append(heur_scores(Y, a, o))
            if opt.get('pf'): Z.append(PF[g.index.values])
            rows.append(pd.DataFrame(np.hstack(Z), index=g.index))
        return pd.concat(rows).loc[df.index]
    PF = pf_matrix(X, opt['pf'], tr_years) if opt.get('pf') else None
    TR = X[X.fy.isin(tr_years)]; VA = X[X.fy == va_year]
    Zt, Zv = design(TR), design(VA); mu, sd = Zt.mean(), Zt.std().replace(0, 1)
    m = LogisticRegression(C=opt['C'], max_iter=5000).fit(((Zt - mu) / sd).values, TR.label.values)
    if all_logits:
        Za = pd.concat([Zt, Zv]); return pd.Series(m.decision_function(((Za - mu) / sd).values), index=Za.index)
    return m.predict_proba(((Zv - mu) / sd).values)[:, 1]

def run_gnn(X, Ys, tr_years, va_year, model, opt, seed):
    """训练一个 GNN（SAGE 或 DFGT 变体）：本折训练年的点对，按验证组内 AUC 早停（patience 15，最多 150 轮）。"""
    torch.manual_seed(seed); np.random.seed(seed)
    years = list(tr_years) + [va_year]; xs = {t: torch.tensor(Ys[t]['x']) for t in years}
    if opt.get('nograph'): Ys = {t: dict(Y, edges=[], deg=np.zeros_like(Y['deg'])) for t, Y in Ys.items()}
    Yp = dict(Ys)   # 用于预计算（消息传递）的节点表；边丢弃只改训练年
    if opt.get('ed'):
        for t in tr_years:
            Y = Ys[t]; rng = np.random.default_rng(20261106 + 100 + seed + 1000 * t); cand = np.where(Y['deg'] > 0)[0]
            dr = np.zeros(len(Y['deg']), bool); dr[rng.choice(cand, int(round(opt['ed'] * len(cand))), replace=False)] = True
            e = [(i, j) for i, j in Y['edges'] if not (dr[i] or dr[j])]; d = np.zeros(len(Y['deg']))
            for i, j in e: d[i] += 1; d[j] += 1
            Yp[t] = dict(Y, edges=e, deg=d)
    if model == 'sage':
        m = SAGE(xs[va_year].shape[1], 128); lr, lam = 1e-2, 0.
        Ps = {t: SAGE.prepare(None, Yp[t]) for t in years}
    else:
        km = fit_kmeans(Ys, years=tr_years)
        Ps = {t: dfgt_prepare_dev(Yp[t], km, zero_k=opt.get('zero_k', 0)) for t in years}
        m = DFGTDev(xs[va_year].shape[1], aug=opt.get('aug', True), use_sim=opt.get('use_sim', True), soft_eval=opt.get('soft_eval', False)); lr, lam = 3e-3, opt.get('lam', 0.1)
    if opt.get('pf'):
        PF = torch.tensor(pf_matrix(X, opt['pf'], tr_years)); m.pf = torch.nn.Linear(PF.shape[1], 1)
        dec = lambda h, u, v, ix: m.dec(h, u, v) + m.pf(PF[ix]).squeeze(-1)
    else: dec = lambda h, u, v, ix: m.dec(h, u, v)
    TR, VA = X[X.fy.isin(tr_years)], X[X.fy == va_year]; PI_tr, PI_va = pair_idx(Ys, TR), pair_idx(Ys, VA)
    y = torch.tensor(X.label.values, dtype=torch.float32); ntr = len(TR)
    opt_ = torch.optim.Adam(m.parameters(), lr=lr); best, state, bad = -1, None, 0
    def score():
        m.eval(); s = np.zeros(len(X))
        with torch.no_grad():
            for t, (u, v, ix) in PI_va.items(): h, _ = m.embed(xs[t], Ps[t]); s[ix] = torch.sigmoid(dec(h, u, v, ix)).cpu().numpy()
        return s[VA.index]
    for ep in range(1, 151):
        m.train(); opt_.zero_grad(); loss = 0.
        for t, (u, v, ix) in PI_tr.items():
            h, ssl = m.embed(xs[t], Ps[t])
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(dec(h, u, v, ix), y[ix], reduction='sum') / ntr + lam * ssl / len(PI_tr)
        loss.backward(); opt_.step(); s = score(); a = fast_auc(VA, s)
        if a > best: best, bad, state = a, 0, s
        else:
            bad += 1
            if bad >= 15: break
    return state

def dcm_inputs(X, N, Ys, years, tr_years, va_year):
    """DCM 的输入：每年的三种邻接矩阵（持股 / 兼任 / 合并）、点对特征（il + port + fine，按本折训练年标准化）、
    点对的持股图度数特征、基础分数（M1 + il 的 logit，只用本折训练年拟合）。"""
    from interlock_feats import interlock_edges
    IE = interlock_edges(); Ps = {}
    for t in years:
        Y = Ys[t]; e = IE.get(t, pd.DataFrame(columns=['src', 'dst'])); idx = Y['idx']
        il = sorted({(min(idx[a], idx[b]), max(idx[a], idx[b])) for a, b in zip(e.src, e.dst) if a in idx and b in idx and a != b})
        Ps[t] = dcm_prepare(len(Y['ids']), Y['edges'], il)
    PF = torch.tensor(pf_matrix(X, ('port', 'fine', 'il'), tr_years))
    DG = np.zeros((len(X), 2), dtype=np.float32)
    for t, g in X[X.fy.isin(years)].groupby('fy'):
        d = Ys[t]['deg']; a = d[[Ys[t]['idx'][c] for c in g.anchor]]; o = d[[Ys[t]['idx'][c] for c in g.other]]
        DG[g.index.values] = np.log1p(np.vstack([np.minimum(a, o), np.maximum(a, o)]).T)
    base = np.zeros(len(X), dtype=np.float32)
    b = run_lr(X, Ys, tr_years, va_year, dict(C=0.1, scores=False, pf=('il',)), all_logits=True); base[b.index.values] = b.values
    return Ps, PF, torch.tensor(DG), torch.tensor(base)

def run_dual(X, Ys, tr_years, va_year, opt, seed, cfg, inputs):
    """训练一个 DCM：本折训练年的点对，按验证组内 AUC 早停（patience 15，最多 150 轮），与 run_gnn 相同。"""
    torch.manual_seed(seed); np.random.seed(seed)
    years = list(tr_years) + [va_year]; xs = {t: torch.tensor(Ys[t]['x']) for t in years}; Ps, PF, DG, base = inputs
    m = DualChannel(xs[va_year].shape[1], PF.shape[1], hid=cfg['hid'], drop=cfg['drop'], rel=opt['rel'], gate=opt['gate'], resid=opt['resid'])
    TR, VA = X[X.fy.isin(tr_years)], X[X.fy == va_year]; PI_tr, PI_va = pair_idx(Ys, TR), pair_idx(Ys, VA)
    y = torch.tensor(X.label.values, dtype=torch.float32); ntr = len(TR)
    opt_ = torch.optim.Adam(m.parameters(), lr=cfg['lr'], weight_decay=1e-5); best, state, bad = -1, None, 0
    lg = lambda h, u, v, ix: m.logit(h, u, v, PF[ix], DG[ix], base[ix])
    def score():
        m.eval(); s = np.zeros(len(X))
        with torch.no_grad():
            for t, (u, v, ix) in PI_va.items(): h = m.embed(xs[t], Ps[t]); s[ix] = torch.sigmoid(lg(h, u, v, ix)).cpu().numpy()
        return s[VA.index]
    for ep in range(1, 151):
        m.train(); opt_.zero_grad(); loss = 0.
        for t, (u, v, ix) in PI_tr.items():
            h = m.embed(xs[t], Ps[t]); loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(lg(h, u, v, ix), y[ix], reduction='sum') / ntr
        loss.backward(); opt_.step(); s = score(); a = fast_auc(VA, s)
        if a > best: best, bad, state = a, 0, s
        else:
            bad += 1
            if bad >= 15: break
    return state, best

def run_base(X, Ys, tr_years, va_year, name, seed, cfg):
    """训练一个新基线：与 run_gnn 相同的训练和早停；DegFairGNN 另加公平正则（系数 1），其他模型的辅助损失系数为 1。"""
    torch.manual_seed(seed); np.random.seed(seed)
    if name == 'seal':
        from seal import run_seal_kr
        return run_seal_kr(X, Ys, tr_years, va_year, seed, cfg, fast_auc)
    years = list(tr_years) + [va_year]; xs = {t: torch.tensor(Ys[t]['x']) for t in years}
    m = BASELINES[name](xs[va_year].shape[1], hid=cfg['hid']); Ps = {t: m.prepare(Ys[t]) for t in years}
    TR, VA = X[X.fy.isin(tr_years)], X[X.fy == va_year]; PI_tr, PI_va = pair_idx(Ys, TR), pair_idx(Ys, VA)
    y = torch.tensor(X.label.values, dtype=torch.float32); ntr = len(TR)
    def score():
        m.eval(); s = np.zeros(len(X))
        with torch.no_grad():
            for t, (u, v, ix) in PI_va.items(): h, _ = m.embed(xs[t], Ps[t]); s[ix] = torch.sigmoid(m.dec(h, u, v)).cpu().numpy()
        return s[VA.index]
    def fit(params, kd_only=False):
        import copy
        opt_ = torch.optim.Adam(params, lr=cfg['lr']); best, state, bad, sd = -1, None, 0, None
        for ep in range(1, 151):
            m.train(); opt_.zero_grad(); loss = 0.
            for t, (u, v, ix) in PI_tr.items():
                h, aux = m.embed(xs[t], Ps[t])
                if kd_only: loss = loss + aux / len(PI_tr); continue
                pl = torch.nn.functional.binary_cross_entropy_with_logits(m.dec(h, u, v), y[ix], reduction='none')
                loss = loss + pl.sum() / ntr + aux / len(PI_tr)
                if name == 'degfair': loss = loss + m.fair_loss(pl, Ps[t], u, v) / len(PI_tr)
            loss.backward(); opt_.step(); s = score(); a = fast_auc(VA, s)
            if a > best: best, bad, state, sd = a, 0, s, copy.deepcopy(m.state_dict())
            else:
                bad += 1
                if bad >= 15: break
        m.load_state_dict(sd); return state, best
    if name == 'coldbrew2':   # 两阶段：先训练教师 + 解码器，再冻结它们只蒸馏学生（第 18.2 节）
        m.stage = 1; fit(m.stage_params()); m.stage = 2; return fit(m.stage_params(), kd_only=True)
    return fit(list(m.parameters()))

def tune_base(name):
    """只用 F1、F2 的验证年为一个基线选超参数：BASE_GRID 的 4 个配置 × 3 个种子，按全部组验证 AUC（分数平均后）两折均值选；全部记录。"""
    X, N, G = load(); rows = []
    for f in ('F1', 'F2'):
        tr_years, va_year = FOLDS[f]; Ys = node_tables(N, G, list(tr_years) + [va_year], tr_years, False); VA = X[X.fy == va_year]
        from seal import SEAL_GRID
        for k, cfg in enumerate(SEAL_GRID if name == 'seal' else BASE_GRID):
            t0 = time.time(); s = np.mean([run_base(X, Ys, tr_years, va_year, name, sd, cfg)[0] for sd in (1, 2, 3)], 0)
            rows.append(dict(model=name, fold=f, cfg=k, **cfg, val_auc=round(fast_auc(VA, s), 4), sec_per_seed=round((time.time() - t0) / 3, 1))); print(rows[-1], flush=True)
    T = pd.DataFrame(rows); T.to_csv(O / f'base_tuning_{name}.csv', index=False); M = T.groupby('cfg').val_auc.mean(); k = int(M.idxmax())
    from seal import SEAL_GRID; GR = SEAL_GRID if name == 'seal' else BASE_GRID
    json.dump(dict(cfg=GR[k], cfg_index=k, mean_val_auc_F1F2=round(float(M.max()), 4)), open(O / f'base_config_{name}.json', 'w'), indent=1); print('selected', name, GR[k])

DCM_GRID = [dict(hid=h, lr=lr, drop=d) for h in (64, 128) for lr in (1e-3, 3e-3) for d in (0.1, 0.3)]   # 事先写定的 8 个配置（第 14 节）

def tune_dcm():
    """只用 F1、F2 的验证年调参：每个配置 3 个种子，按“全部组”验证组内 AUC（分数平均后）的两折均值选；全部配置写 dcm_tuning.csv，选中的写 dcm_config.json。"""
    X, N, G = load(); rows = []
    for f in ('F1', 'F2'):
        tr_years, va_year = FOLDS[f]; years = list(tr_years) + [va_year]
        Ys = node_tables(N, G, years, tr_years, False); inp = dcm_inputs(X, N, Ys, years, tr_years, va_year); VA = X[X.fy == va_year]
        for k, cfg in enumerate(DCM_GRID):
            s = np.mean([run_dual(X, Ys, tr_years, va_year, VARIANTS['dcm'][1], sd, cfg, inp)[0] for sd in (1, 2, 3)], 0)
            rows.append(dict(fold=f, cfg=k, **cfg, val_auc=round(fast_auc(VA, s), 4))); print(rows[-1], flush=True)
    T = pd.DataFrame(rows); T.to_csv(O / 'dcm_tuning.csv', index=False)
    M = T.groupby('cfg').val_auc.mean(); k = int(M.idxmax()); cfg = DCM_GRID[k]
    json.dump(dict(cfg=cfg, cfg_index=k, mean_val_auc_F1F2=round(float(M.max()), 4), rule='mean all-group val AUC on F1, F2; 3 seeds'), open(O / 'dcm_config.json', 'w'), indent=1)
    print('selected', k, cfg, round(float(M.max()), 4)); print(M.round(4).to_string())

def run(variant):
    """跑一个变体：3 折 ×（逻辑回归 1 次 / GNN 3 个种子取平均分数），保存验证分数。"""
    model, opt = VARIANTS[variant]; X, N, G = load(); rows = []; RT.clear()
    if model == 'ens':
        S = [pd.read_csv(O / sfile(p), dtype={'anchor': str, 'other': str, 'pair_id': str}) for p in opt['parts']]
        out = S[0].drop(columns='score').assign(score=np.mean([s_.groupby('fy').score.rank(pct=True).values for s_ in S], 0))
        out.to_csv(sfile(variant), index=False); print(variant, 'written'); return
    if opt.get('ile'):   # 把 FY t 的兼任边并入 G_t（只保留两端都是当年合格企业的边，不新增节点）；评估时的度分组仍按持股图
        from interlock_feats import interlock_edges
        IE = interlock_edges(); G = dict(G)
        for t in IE:
            if t not in G: continue
            el = set(N[N.fy == t].corp_code); e = IE[t]
            if opt['ile'] == 'inside': e = e[e.inside == 1]
            e = e[e.src.isin(el) & e.dst.isin(el)]
            G[t] = pd.concat([G[t], e[['src', 'dst']]], ignore_index=True)
            print('FY', t, 'interlock edges added', len(e), flush=True)
    for f, (tr_years, va_year) in FOLDS.items():
        Ys = node_tables(N, G, list(tr_years) + [va_year], tr_years, opt.get('target', False)); VA = X[X.fy == va_year]
        per = {}
        if model == 'lr': s = run_lr(X, Ys, tr_years, va_year, opt)
        elif model == 'base':
            cfg = json.load(open(O / f"base_config_{opt.get('cfg_from', opt['name'])}.json"))['cfg']
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            t0 = time.time(); per = {f'score_s{sd}': run_base(X, Ys, tr_years, va_year, opt['name'], sd, cfg)[0] for sd in SEEDS}; s = np.mean(list(per.values()), 0)
            RT.append(dict(model=variant, fold=f, sec_per_seed=round((time.time() - t0) / len(SEEDS), 2), gpu_mb=round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0))
        elif model == 'dcm':
            cfg = json.load(open(O / 'dcm_config.json'))['cfg']; years = list(tr_years) + [va_year]; inp = dcm_inputs(X, N, Ys, years, tr_years, va_year)
            per = {f'score_s{sd}': run_dual(X, Ys, tr_years, va_year, opt, sd, cfg, inp)[0] for sd in SEEDS}; s = np.mean(list(per.values()), 0)
        else:
            per = {f'score_s{sd}': run_gnn(X, Ys, tr_years, va_year, model, opt, sd) for sd in SEEDS}; s = np.mean(list(per.values()), 0)
        rows.append(VA[['pair_id', 'fy', 'anchor', 'other', 'label']].assign(fold=f, score=s, **(per if len(SEEDS) > 3 else {})))   # 多种子时另存每个种子的分数
        print(variant, f, round(fast_auc(VA, s), 4), flush=True)
    pd.concat(rows).to_csv(sfile(variant), index=False)
    if RT:
        f = O / 'runtime_kr.csv'; old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=['model'])
        pd.concat([old[old.model != variant], pd.DataFrame(RT)]).to_csv(f, index=False)

def evaluate():
    """汇总：每个正例的组内 AUC；按较小端度数分低连接（≤1）/ 高连接（≥2）/ 孤立（=0）；差距和配对 bootstrap（按 折×投资方 聚类）。"""
    X, N, G = load(); deg = {t: collections.Counter(list(G[t].src) + list(G[t].dst)) for t in G}
    W = None
    for v in VARIANTS:
        f = sfile(v)
        if not f.exists(): continue
        S = pd.read_csv(f, dtype={'anchor': str, 'other': str, 'pair_id': str}); S['pos_group'] = ''
        w = within_metrics(S, 'score')[['AUC', 'H10']].rename(columns={'AUC': f'{v}|AUC', 'H10': f'{v}|H10'})
        if W is None:
            pos = S[S.label == 1].set_index('pair_id'); W = w.join(pos[['anchor', 'fold', 'fy', 'other']])
            W['mindeg'] = [min(deg[t][a], deg[t][o]) for t, a, o in zip(W.fy, W.anchor, W.other)]
            W['cluster'] = W.fold + '-' + W.anchor
        else: W = W.join(w)
    vs = [v for v in VARIANTS if f'{v}|AUC' in W]; W = W.reset_index()
    grp = {'all': W.index == W.index, 'low(≤1)': W.mindeg <= 1, 'high(≥2)': W.mindeg >= 2, 'isolated(0)': W.mindeg == 0}
    tab = pd.DataFrame({g: {v: W.loc[m, f'{v}|AUC'].mean() for v in vs} for g, m in grp.items()}); tab['gap(high-low)'] = tab['high(≥2)'] - tab['low(≤1)']
    tab['n'] = ''; ns = {g: int(m.sum()) for g, m in grp.items()}
    cl = W.cluster.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(20261103)
    draws = [np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]) for _ in range(A.boot)]   # 默认 1000；确认时 P1、C1、C2 用 --boot 10000
    def ci(fn):
        v = np.array([fn(W.iloc[d]) for d in draws]); p = min(1., 2 * min(float(np.mean(v <= 0)), float(np.mean(v >= 0))))   # 双侧 bootstrap p 值（分辨率 2/B）
        return float(fn(W)), float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)), float(np.nanpercentile(v, 5)), p   # (点估计, 双侧 95% 下限, 上限, 单侧 95% 下限, 双侧 p)；不在这里四舍五入，避免打印时二次舍入
    mean = lambda v, g: (lambda U: U.loc[(U.mindeg <= 1) if g == 'low' else (U.mindeg >= 2) if g == 'high' else (U.mindeg == 0) if g == 'iso' else U.index == U.index, f'{v}|AUC'].mean())
    gap = lambda v: (lambda U: mean(v, 'high')(U) - mean(v, 'low')(U))
    comps = {}
    for a, b in [('dfgt', 'M1'), ('dfgt', 'sage'), ('dfgt', 'dfgt-noaug'), ('dfgt', 'dfgt-nosim'), ('dfgt', 'dfgt-nossl'), ('dfgt0', 'dfgt'), ('dfgt-soft', 'dfgt'), ('dfgt-soft', 'dfgt-noaug'), ('dfgt0-soft', 'dfgt-soft'), ('dfgt-soft', 'M1'), ('dfgt-soft', 'sage'), ('dfgt-soft', 'dfgt-soft-nossl'), ('dfgt', 'dfgt-nograph'), ('dfgt-nograph', 'M1'),
                 ('dfgt+T', 'dfgt'), ('dfgt0+T', 'dfgt'), ('M1+T', 'M1'), ('sage+T', 'sage'), ('M1+scores', 'M1'),
                 ('M1+ind', 'M1'), ('M1+port', 'M1'), ('M1+pf', 'M1'), ('M1+fine', 'M1'), ('M1+hist', 'M1'), ('M1+port+fine', 'M1'), ('M1+il', 'M1'), ('M1+port+il', 'M1+port'), ('M1+pf+il', 'M1+pf'), ('M1+il-lenient', 'M1'), ('dfgt+il', 'dfgt'), ('dfgt+pf+il', 'dfgt'), ('dfgt-nograph+il', 'dfgt-nograph'), ('dfgt+il', 'M1+il'), ('dfgt+ile', 'dfgt'), ('dfgt+ile+il', 'dfgt'), ('dfgt+ile+il', 'dfgt+il'), ('dfgt+ile+pf+il', 'dfgt'), ('dfgt+ile_in', 'dfgt'), ('sage+ile', 'sage'), ('dfgt+ile', 'M1+il'), ('ens:dfgt|M1+il', 'dfgt'), ('ens:dfgt|M1+il', 'ens:dfgt|M1'), ('ens:dfgt|M1', 'dfgt'), ('dfgt+port+fine+il', 'dfgt+port+fine'), ('dfgt+port+fine', 'dfgt'), ('M1+port+fine+il', 'M1+port+fine'), ('dfgt+pf', 'dfgt+port+fine'), ('M1+hs', 'M1'), ('gae', 'dfgt'), ('gae', 'dfgt-nograph'), ('gae', 'sage'), ('coldbrew', 'dfgt'), ('coldbrew', 'dfgt-nograph'), ('coldbrew', 'sage'), ('degfair', 'dfgt'), ('degfair', 'dfgt-nograph'), ('degfair', 'sage'), ('tailgnn', 'dfgt'), ('tailgnn', 'dfgt-nograph'), ('tailgnn', 'sage'), ('dcm', 'dfgt+port+fine+il'), ('dcm', 'M1+il'), ('dcm', 'dcm-noA'), ('dcm', 'dcm-noB'), ('dcm', 'dcm-noC'), ('dcm-noA', 'dfgt+port+fine+il'), ('dcm-noB', 'dfgt+port+fine+il'), ('dcm-noC', 'dfgt+port+fine+il'), ('M1+car', 'M1'), ('M1+il+car', 'M1+il'), ('M1+ms', 'M1'), ('M1+il+ms', 'M1+il'), ('dfgt+port+fine+il+car', 'dfgt+port+fine+il'), ('dfgt+port+fine+il+ms', 'dfgt+port+fine+il'), ('dfgt+port+fine+il+hs', 'dfgt+port+fine+il'), ('M1+port+fine+il+hs', 'M1+port+fine+il'), ('M1+il+hs', 'M1+il'), ('M1+pf+il+hs', 'M1+pf+il'), ('M1+hs-lenient', 'M1'), ('dfgt+pf+hs', 'dfgt+pf'), ('dfgt+pf+il+hs', 'dfgt+pf+il'), ('dfgt+ile+pf+il+hs', 'dfgt+ile+pf+il'), ('dfgt+pf+il+hs', 'dfgt'), ('dfgt+pf+il', 'dfgt+pf'), ('dfgt+ile+pf+il', 'dfgt+pf+il'), ('dfgt+pf+il', 'M1+pf+il'), ('dfgt+pf', 'dfgt'), ('dfgt-nograph+pf', 'dfgt-nograph'), ('dfgt+pf', 'M1+pf'),
                 ('dfgt-ed10', 'dfgt'), ('dfgt-ed30', 'dfgt'), ('dfgt-ed10', 'dfgt-nograph'), ('dfgt-ed30', 'dfgt-nograph'), ('sage-ed10', 'sage'), ('sage-ed30', 'sage'), ('coldbrew2', 'dfgt'), ('coldbrew2', 'dfgt-nograph'), ('coldbrew2', 'coldbrew'), ('seal', 'dfgt'), ('seal', 'dfgt-nograph'), ('seal', 'sage'), ('seal', 'M1')]:   # 第 18 节 C1–C3
        if a in vs and b in vs and (not A.only or f'{a}:{b}' in A.only.split(',')):
            comps[f'{a} − {b}'] = {g: ci(lambda U, a=a, b=b, g=g: mean(a, g)(U) - mean(b, g)(U)) for g in ('all', 'low', 'high', 'iso')}
            comps[f'{a} − {b}']['gap_reduction'] = ci(lambda U, a=a, b=b: gap(b)(U) - gap(a)(U))
    gaps = {v: ci(gap(v)) for v in vs if not A.only or v in A.gaps.split(',')}   # 每个模型自身的度偏差差距 Δ = AUC(高连接) − AUC(低连接)，同一套 bootstrap 抽样
    r4 = lambda x: [round(v, 4) for v in x]
    res = dict(n=ns, table=tab.drop(columns='n').round(4).to_dict('index'), gap_ci={k: r4(x) for k, x in gaps.items()},
               comparisons={k: {g: r4(x) for g, x in v.items()} for k, v in comps.items()})
    json.dump(res, open(O / (A.save or ('dev_eval.json' if A.boot == 1000 and not A.only else f'dev_eval_boot{A.boot}.json')), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('n', ns); print(tab.drop(columns='n').round(3).to_string())
    print('Δ (high − low) [95%]:'); [print(f'  {v:22s} {x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]') for v, x in gaps.items()]
    for k, v in comps.items(): print(f'{k:22s}', '  '.join(f'{g}: {x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]' for g, x in v.items()))
    if 'dfgt' in gaps and 'dfgt − M1' in comps:   # 确认性假设 C1、C2 族（3 个假设）的 Holm 判定；满足 = Holm 显著且点估计方向为正
        fam = {'C1 Δ(dfgt)': gaps['dfgt'], 'C2 dfgt − M1 (all)': comps['dfgt − M1']['all'], 'C2 dfgt − M1 (low)': comps['dfgt − M1']['low']}
        order = sorted(fam, key=lambda k: fam[k][4]); stop = False; res['holm_C1_C2'] = {}
        print(f'Holm（C1、C2 族，B={A.boot}）：')
        for r, k in enumerate(order):
            thr = .05 / (len(order) - r); sig = (not stop) and fam[k][4] < thr; stop = stop or not sig
            verdict = ('满足' if fam[k][0] > 0 else '不满足（方向相反）') if sig else '不满足'
            res['holm_C1_C2'][k] = dict(est=round(fam[k][0], 4), p=fam[k][4], threshold=round(thr, 4), verdict=verdict)
            print(f'  {k:20s} est {fam[k][0]:+.3f}  p {fam[k][4]:.4f}  阈值 {thr:.4f}  → {verdict}')
        json.dump(res, open(O / (A.save or ('dev_eval.json' if A.boot == 1000 and not A.only else f'dev_eval_boot{A.boot}.json')), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('单侧 95% 下限（5% 分位）：'); [print(f'  {k:22s}', '  '.join(f'{g}: {x[3]:+.3f}' for g, x in v.items())) for k, v in comps.items() if k in ('M1+il − M1', 'dfgt − M1')]

ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--variant'); ap.add_argument('--threads', type=int, default=3)
ap.add_argument('--device', default='cpu', help="cpu（默认）或 cuda；cuda 时所有张量默认建在 GPU 上，模型代码不用改"); ap.add_argument('--seeds', type=int, default=3, help='GNN 种子数（种子为 1..n；默认 3，与之前一致）')
ap.add_argument('--boot', type=int, default=1000, help='bootstrap 次数（默认 1000；确认性检验 P1、C1、C2 用 10000）')
ap.add_argument('--only', default='', help='只算这些比较，逗号分隔的 a:b（例如 M1+il:M1）；配合 --boot 使用，结果写 dev_eval_boot<B>.json')
ap.add_argument('--gaps', default='', help='与 --only 一起用：只算这些模型自身的 Δ 区间，逗号分隔')
ap.add_argument('--save', default='', help='evaluate 的输出文件名（覆盖默认命名，例如 dev_eval_c.json）')
ap.add_argument('--out', default='out_dev', help='分数输出目录（相对 linkpred/）'); A = ap.parse_args()
O = L / A.out; O.mkdir(exist_ok=True); SEEDS = list(range(1, A.seeds + 1))
if torch is not None:
    torch.set_num_threads(A.threads)
    if A.device != 'cpu': torch.set_default_device(A.device)
if A.cmd == 'run': run(A.variant)
elif A.cmd == 'eval': evaluate()
elif A.cmd == 'tune-dcm': tune_dcm()
elif A.cmd == 'tune-base': tune_base(A.variant)
