"""DEV_RESULTS 第 21 节：第二轮修订分析（只用开发年做比较；不改标签、不改冻结代码）。
  A  正例的首次取得日与预测日 C_t（t+1 年 4 月 30 日）：按类别重算 P1、S_iso、C3′、DFGT − M1（第 21.1 节）
  B  正例标签证据的四分类（第 21.2 节）
  C  单个模型的数字和种子配对差（第 21.3 节）
  D  度的计数口径（第 21.4 节）
用法：python src/label_audit.py [--scratch D:/claude\ data/scratch_label_audit]
出资行由本进程内执行 build_v3.py 的代码重建（--first-year 2019，不调用 API，输出写到 --scratch，不碰 data_v3x），
并先核对重建的 pairs.csv 与 data_v3x/pairs.csv 逐字节相同。结果写 out_dev_gpu10/label_audit.json。"""
import argparse, collections, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
L = Path(__file__).resolve().parents[1]; SRC = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--scratch', default=str(L.parent.parent / 'scratch_label_audit')); A = ap.parse_args()
SC = Path(A.scratch); SC.mkdir(parents=True, exist_ok=True)
SD = L / 'out_dev_gpu10'; DATA = L / 'data_v3x'; DEV = (2021, 2022, 2023)   # 三折的验证年 = 575 个评估正例
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
OUT = {}

# ---------- 重建出资行：执行 build_v3.py 的代码 ----------
argv0 = sys.argv; sys.argv = ['build_v3.py', '--out', str(SC), '--first-year', '2019']
ns = dict(__name__='__main__', __file__=str(SRC / 'build_v3.py'))
exec(compile((SRC / 'build_v3.py').read_text(encoding='utf-8'), str(SRC / 'build_v3.py'), 'exec'), ns)
sys.argv = argv0
same = sha(SC / 'pairs.csv') == sha(DATA / 'pairs.csv')
OUT['rebuild_pairs_identical'] = same; print('rebuilt pairs.csv identical to data_v3x:', same, flush=True)
if not same: sys.exit('重建的 pairs.csv 与 data_v3x 不一致，停止（第 21.1 节）')
ROWS, parse_date, num = ns['ROWS'], ns['parse_date'], ns['num']
X = pd.read_csv(DATA / 'pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str, 'directions': str})
POS = X[X.label == 1].set_index('pair_id')

def fy_bounds(stlm, fy1):
    import re
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', stlm or ''); end = tuple(map(int, m.groups())) if m else (fy1, 12, 31)
    return (end[0] - 1, end[1], end[2]), end

def row_date_cat(acq, stlm, t):
    """第 21.1 节的行类别。"""
    fy1 = t + 1; start, end = fy_bounds(stlm, fy1); cut = (fy1, 4, 30); d = parse_date(acq)
    if d is None: return 'missing'
    if d[1] is None:
        return 'year_only' if d[0] == fy1 else ('earlier' if d[0] < fy1 else 'later')
    if d <= start: return 'earlier'
    if d > end: return 'later'
    return 'before' if d <= cut else 'after'

def pair_rows(pid):
    p = POS.loc[pid]; t = int(p.fy); R1 = ROWS[t + 1]; out = []
    for dr in str(p.directions).split('+'):
        a, b = dr.split('>'); out.append(R1[(R1.src_corp_code == a) & (R1.t == b)])
    return t, pd.concat(out)

rec = []
for pid in POS.index:
    t, rr = pair_rows(pid)
    dc = [row_date_cat(a, s, t) for a, s in zip(rr.frst_acqs_de, rr.stlm_dt)]
    if 'before' in dc: A_cat = 'before'
    elif 'after' in dc: A_cat = 'after'
    else: A_cat = 'unknown:' + next(c for c in ('earlier', 'later', 'year_only', 'missing') if c in dc)
    # B：四分类
    oq, ob, orr = num(rr.bsis_blce_qy), num(rr.bsis_blce_acntbk_amount), num(rr.bsis_blce_qota_rt)
    base, conflict = [], []
    for i in range(len(rr)):
        vals = [v for v in (oq.iloc[i], ob.iloc[i], orr.iloc[i]) if pd.notna(v)]
        b = '2_acq_date' if bool(rr.open_pos.iloc[i]) else ('1_explicit_zero' if vals and all(v == 0 for v in vals) else '3_all_missing')
        base.append(b); conflict.append(b != '2_acq_date' and dc[i] == 'earlier')
    B_cat = '4_conflict' if any(conflict) else next(c for c in ('1_explicit_zero', '2_acq_date', '3_all_missing') if c in base)
    in_year = any(c in ('before', 'after') for c in dc)
    rec.append(dict(pair_id=pid, fy=t, n_rows=len(rr), A=A_cat, A_main=A_cat.split(':')[0], B=B_cat, dates='|'.join(sorted(set(dc))),
                    B3_date=('in_year' if in_year else next((c for c in ('year_only', 'missing', 'earlier', 'later') if c in dc), 'none')), new_evidence=POS.loc[pid].new_evidence))
R = pd.DataFrame(rec).set_index('pair_id'); R.to_csv(SC / 'label_audit_pairs.csv')
assert (R.n_rows > 0).all(), '有正例没有对应的出资行'

# 度（冻结评估代码的口径：有向 graph_t 上 Counter(src + dst)）与无向不重复邻居
G = {t: pd.read_csv(DATA / f'graph_{t}.csv', dtype=str) for t in range(2019, 2025)}
deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t, g in G.items()}
udeg = {}
for t, g in G.items():
    nb = collections.defaultdict(set)
    for a, b in zip(g.src, g.dst): nb[a].add(b); nb[b].add(a)
    udeg[t] = {k: len(v) for k, v in nb.items()}
R['mindeg'] = [min(deg[t][POS.loc[p].anchor], deg[t][POS.loc[p].other]) for p, t in zip(R.index, R.fy)]
R['mindeg_u'] = [min(udeg[t].get(POS.loc[p].anchor, 0), udeg[t].get(POS.loc[p].other, 0)) for p, t in zip(R.index, R.fy)]
dev = R[R.fy.isin(DEV)]

# ---------- B ----------
OUT['B'] = dict(by_year=pd.crosstab(R.fy, R.B, margins=True).to_dict('index'), dev_total=dev.B.value_counts().to_dict(),
                dev_low=dev[dev.mindeg <= 1].B.value_counts().to_dict(), dev_iso=dev[dev.mindeg == 0].B.value_counts().to_dict(),
                cat3_by_date_dev=dev[dev.B == '3_all_missing'].B3_date.value_counts().to_dict(),
                cat3_by_date_by_year=pd.crosstab(R[R.B == '3_all_missing'].fy, R[R.B == '3_all_missing'].B3_date).to_dict('index'),
                vs_new_evidence_dev=pd.crosstab(dev.B, dev.new_evidence).to_dict('index'))
v2 = {'2021': (260, 10, 69, 4, 3), '2022': (290, 6, 79, 6, 1), '2023': (303, 62, 69, 17, 0), '2024': (339, 0, 52, 11, 2)}   # reports/DATA_V3.md 第 1 节
OUT['B']['table2_shares'] = {y: dict(naive=n, name_change=round(a / n, 4), held_not_listed=round(b / n, 4), inconsistent=round(c / n, 4), filters=round(d / n, 4), total=round((a + b + c + d) / n, 4)) for y, (n, a, b, c, d) in v2.items()}

# ---------- D ----------
OUT['D'] = dict(definition='paper_dev.evaluate: Counter(list(G[t].src) + list(G[t].dst)) on directed graph_t.csv; reciprocal holdings count twice',
                dev_low_counter=int((dev.mindeg <= 1).sum()), dev_low_unique=int((dev.mindeg_u <= 1).sum()),
                dev_iso_counter=int((dev.mindeg == 0).sum()), dev_iso_unique=int((dev.mindeg_u == 0).sum()), dev_n=len(dev),
                reciprocal_pairs_by_year={t: sum(1 for a, b in set(zip(g.src, g.dst)) if a < b and (b, a) in set(zip(g.src, g.dst))) for t, g in G.items()})

# ---------- 每个正例的组内 AUC ----------
def per_pos(f, col='score'):
    S = pd.read_csv(f, dtype={'pair_id': str, 'anchor': str, 'other': str}).sort_values(['pair_id', 'label'], ascending=[True, False])
    cols = [c for c in S.columns if c.startswith('score')] if col == 'all' else [col]
    out = {}
    for pid, g in S.groupby('pair_id', sort=False):
        lab = g.label.values
        for c in cols:
            v = g[c].values; s = v[lab == 1][0]; n = v[lab == 0]
            out.setdefault(c, {})[pid] = ((n < s).sum() + .5 * (n == s).sum()) / len(n)
    return pd.DataFrame(out)
S0 = pd.read_csv(SD / 'scores_M1.csv', dtype={'pair_id': str, 'anchor': str}); fold = S0[S0.label == 1].set_index('pair_id').fold
dev = dev.assign(cluster=fold.reindex(dev.index) + '-' + POS.loc[dev.index].anchor)
GRP = {'all': lambda U: np.ones(len(U), bool), 'low': lambda U: (U.mindeg <= 1).values, 'high': lambda U: (U.mindeg >= 2).values, 'iso': lambda U: (U.mindeg == 0).values}
def draws_for(U, B, seed=20261103):
    cl = U.cluster.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(seed)
    return [np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]) for _ in range(B)]
def ci(d, m, dr):
    v = np.array([d[x][m[x]].mean() for x in dr]); return [round(float(d[m].mean()), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4), round(float(np.percentile(v, 5)), 4), int(m.sum())]

AUC = {v: per_pos(SD / f'scores_{v}.csv').score.reindex(dev.index) for v in ('M1', 'M1+il', 'dfgt', 'dfgt+port+fine', 'dfgt+port+fine+il')}
COMPS = [('M1+il', 'M1', 10000), ('dfgt+port+fine+il', 'dfgt+port+fine', 1000), ('dfgt', 'M1', 1000)]
OUT['A'] = dict(counts={g: dev[GRP[g](dev)].A.value_counts().to_dict() for g in ('all', 'low', 'iso')},
                counts_main={g: dev[GRP[g](dev)].A_main.value_counts().to_dict() for g in ('all', 'low', 'iso')},
                by_year=pd.crosstab(dev.fy, dev.A_main).to_dict('index'), results={})
subsets = {'full': np.ones(len(dev), bool), 'after': (dev.A_main == 'after').values, 'before': (dev.A_main == 'before').values, 'unknown': (dev.A_main == 'unknown').values}
for sname, sm in subsets.items():
    U = dev[sm]; DR = {B: draws_for(U, B) for B in (1000, 10000)}
    for a, b, B in COMPS:
        d = (AUC[a] - AUC[b])[sm].values
        OUT['A']['results'].setdefault(sname, {})[f'{a} − {b}'] = {g: ci(d, GRP[g](U), DR[B]) for g in ('all', 'low', 'iso')}
# before − after（两个子集各自独立重抽）
Ua, Ub = dev[subsets['after']], dev[subsets['before']]; da_, db_ = draws_for(Ua, 10000, 20261103), draws_for(Ub, 10000, 20261104)
OUT['A']['before_minus_after'] = {}
for a, b, B in COMPS:
    xa = (AUC[a] - AUC[b])[subsets['after']].values; xb = (AUC[a] - AUC[b])[subsets['before']].values; R_ = {}
    for g in ('all', 'low', 'iso'):
        ma, mb = GRP[g](Ua), GRP[g](Ub); v = np.array([xb[y][mb[y]].mean() - xa[x][ma[x]].mean() for x, y in zip(da_[:B], db_[:B])])
        R_[g] = [round(float(xb[mb].mean() - xa[ma].mean()), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    OUT['A']['before_minus_after'][f'{a} − {b}'] = R_
full = OUT['A']['results']['full']['M1+il − M1']['low'][0]; aft = OUT['A']['results']['after']['M1+il − M1']['low']
OUT['A']['reading'] = dict(full_low=full, after_low=aft[0], retained=round(aft[0] / full, 4), one_sided_lower=aft[3],
                           verdict='稳健' if aft[3] > 0 and aft[0] >= .75 * full else ('方向一致但减弱' if aft[3] > 0 else '不稳健'))

# ---------- C ----------
MODELS = ['dfgt', 'dfgt-nograph', 'sage', 'dfgt+port+fine', 'dfgt+port+fine+il', 'dfgt-tuned', 'gae', 'coldbrew', 'coldbrew2', 'degfair', 'tailgnn', 'seal', 'tailgnn-official', 'seal-official']
m = {g: GRP[g](dev) for g in GRP}
def grp_auc(s): r = {g: float(s[m[g]].mean()) for g in m}; r['gap'] = r['high'] - r['low']; return r
C = {'M1': grp_auc(AUC['M1'].values), 'M1+il': grp_auc(AUC['M1+il'].values)}; PS = {}
for v in MODELS:
    P = per_pos(SD / f'scores_{v}.csv', 'all').reindex(dev.index); PS[v] = P
    seeds = [c for c in P.columns if c != 'score']; per = pd.DataFrame([grp_auc(P[c].values) for c in seeds])
    C[v] = dict(ensemble={k: round(x, 4) for k, x in grp_auc(P.score.values).items()}, n_seeds=len(seeds),
                single_mean={k: round(float(per[k].mean()), 4) for k in per}, single_sd={k: round(float(per[k].std(ddof=1)), 4) for k in per})
OUT['C'] = dict(models=C, paired={})
for a, b in (('dfgt', 'M1'), ('dfgt-nograph', 'M1'), ('dfgt+port+fine+il', 'dfgt+port+fine')):
    seeds = [c for c in PS[a].columns if c != 'score']
    D_ = pd.DataFrame([{k: grp_auc(PS[a][c].values)[k] - (C[b][k] if b in ('M1', 'M1+il') else grp_auc(PS[b][c].values)[k]) for k in ('all', 'low', 'high', 'iso', 'gap')} for c in seeds])
    OUT['C']['paired'][f'{a} − {b}'] = {k: dict(mean=round(float(D_[k].mean()), 4), sd=round(float(D_[k].std(ddof=1)), 4), n_pos=int((D_[k] > 0).sum()), n=len(D_)) for k in D_}
json.dump(OUT, open(SD / 'label_audit.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: OUT[k] for k in ('D',)}, ensure_ascii=False, indent=1))
print(json.dumps(OUT['A']['counts'], ensure_ascii=False)); print(json.dumps(OUT['A']['reading'], ensure_ascii=False))
for s, v in OUT['A']['results'].items():
    for k, x in v.items(): print(f'{s:8s} {k:36s}', '  '.join(f'{g}: {y[0]:+.3f} [{y[1]:+.3f},{y[2]:+.3f}] lo{y[3]:+.3f} n{y[4]}' for g, y in x.items()))
print('before − after', json.dumps(OUT['A']['before_minus_after'], ensure_ascii=False))
print('B by year', json.dumps(OUT['B']['by_year'], ensure_ascii=False)); print('B cat3 dates', OUT['B']['cat3_by_date_dev'], OUT['B']['vs_new_evidence_dev'])
for v, x in OUT['C']['models'].items():
    if 'ensemble' in x: print(f'{v:20s} ens', x['ensemble'], 'single', x['single_mean'], 'sd', x['single_sd'])
    else: print(f'{v:20s}', {k: round(y, 4) for k, y in x.items()})
for k, x in OUT['C']['paired'].items(): print(k, x)
