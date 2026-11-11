"""DEV_RESULTS 第 25.6 节：企业集团归属对照（公平交易委员会名单；只用开发年；只用逻辑回归；不改冻结代码）。
用法：python src/group_control.py   # 写 out_dev_group/、out_dev_rewire/group_eval.json"""
import collections, contextlib, io, json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
L = Path(__file__).resolve().parents[1]; P0 = L.parent; RW = L / 'out_dev_rewire'; GD = L / 'out_dev_group'; GD.mkdir(exist_ok=True)
argv0 = sys.argv; sys.argv = ['paper_dev.py', 'none', '--out', 'out_dev_group', '--device', 'cpu']
import paper_dev as PD
sys.argv = argv0
A = pd.read_csv(P0 / 'data/interim/kftc_affiliates_2019_2025.csv', dtype=str); O = pd.read_csv(P0 / 'data/interim/company_overview.csv', dtype=str, keep_default_na=False)
j = dict(zip(O.jurir_no.str.replace('-', '').str.strip(), O.corp_code)); A['corp_code'] = A['법인등록번호'].str.replace('-', '').str.strip().map(j)
grp = collections.defaultdict(set)   # (年, 公司) → 集团名集合
for y, c, g in zip(A.appl_year.astype(int), A.corp_code, A['기업집단명']):
    if isinstance(c, str): grp[(y, c)].add(g)
X = pd.read_csv(L / 'data_v3x' / 'pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str}); X = X[X.fy.between(2019, 2023)]
ga = [grp.get((t, a), set()) for t, a in zip(X.fy, X.anchor)]; go = [grp.get((t, o), set()) for t, o in zip(X.fy, X.other)]
same = np.array([bool(x & y) for x, y in zip(ga, go)]); ina = np.array([bool(x) for x in ga]); ino = np.array([bool(y) for y in go])
cat = np.where(same, 'SG', np.where(ina & ino, 'DG', np.where(ina | ino, 'ONE', 'NONE')))
for f in ('pair_feats.csv', 'hyslr_feats.csv', 'interlock_feats.csv'): shutil.copyfile(RW / 'real' / f, GD / f)
pd.DataFrame({'ms_common': same.astype(float), 'ms_common_inst': ina.astype(float), 'ms_common_indiv': ino.astype(float), 'ms_has_a': 0., 'ms_has_o': 0.}, index=X.index).to_csv(GD / 'majorstock_feats.csv', index_label='row')
pd.DataFrame({'cat': cat}, index=X.index).to_csv(GD / 'group_cat.csv', index_label='row')
PD.FE = GD; PD.O = GD
with contextlib.redirect_stdout(io.StringIO()):
    for v in ('M1+ms', 'M1+il+ms'): PD.run(v)

deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t in (2021, 2022, 2023) for g in [pd.read_csv(L / f'data_v3x/graph_{t}.csv', dtype=str)]}
def per_pos(f):
    S = pd.read_csv(f, dtype={'pair_id': str, 'anchor': str, 'other': str}); r = {}
    for pid, g in S.groupby('pair_id'):
        p = g[g.label == 1].iloc[0]; n = g.score.values[g.label.values == 0]; s = p.score
        r[pid] = (p.fold, p.anchor, min(deg[p.fy][p.anchor], deg[p.fy][p.other]), ((n < s).sum() + .5 * (n == s).sum()) / len(n))
    return pd.DataFrame(r, index=['fold', 'anchor', 'md', 'auc']).T
W = per_pos(RW / 'real' / 'scores_M1.csv'); W['md'] = W.md.astype(int); W['cl'] = W.fold + '-' + W.anchor; W['M1'] = W.auc.astype(float)
W['M1+il'] = per_pos(RW / 'real' / 'scores_M1+il.csv').auc.astype(float); W['G'] = per_pos(GD / 'scores_M1+ms.csv').auc.astype(float); W['G+il'] = per_pos(GD / 'scores_M1+il+ms.csv').auc.astype(float)
pos = X[X.label == 1].assign(cat=cat[(X.label == 1).values]).set_index('pair_id'); W['cat'] = pos.cat.reindex(W.index)
il = pd.read_csv(RW / 'real' / 'interlock_feats.csv', index_col='row'); sig = (il[['il_direct', 'il_inside', 'il_2hop', 'il_hist']] > 0).any(axis=1)
W['sig'] = X[X.label == 1].assign(s=sig.reindex(X[X.label == 1].index).values).set_index('pair_id').s.reindex(W.index).values
cl = W.cl.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(20261103); DR = [np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]) for _ in range(1000)]
def ci(d, m):
    if m.sum() < 5: return None
    v = [d[i][m[i]].mean() for i in DR if m[i].any()]
    return [round(float(d[m].mean()), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4), int(m.sum())]
GR = {'low': (W.md <= 1).values, 'iso': (W.md == 0).values, 'all': np.ones(len(W), bool)}
gain = (W['M1+il'] - W['M1']).values; subs = {'SG': (W.cat == 'SG').values, 'not SG': (W.cat != 'SG').values, 'DG': (W.cat == 'DG').values, 'ONE': (W.cat == 'ONE').values, 'NONE': (W.cat == 'NONE').values}
res = dict(counts={g: W[m].cat.value_counts().to_dict() for g, m in GR.items()},
           interlock_signal_share_by_cat={k: round(float(W[W.cat == k].sig.mean()), 4) for k in ('SG', 'DG', 'ONE', 'NONE') if (W.cat == k).any()},
           SG_share_among_signal=round(float((W[W.sig.astype(bool)].cat == 'SG').mean()), 4), n_signal=int(W.sig.astype(bool).sum()),
           il_beyond_G={g: ci((W['G+il'] - W['G']).values, m) for g, m in GR.items()}, G_alone={g: ci((W['G'] - W['M1']).values, m) for g, m in GR.items()},
           il_gain_by_subset={s: {g: ci(gain, m & sm) for g, m in GR.items()} for s, sm in subs.items()})
lo = res['il_beyond_G']['low']; ns = res['il_gain_by_subset']['not SG']['low']
res['reading'] = '兼任信号不只是同集团的代理' if lo and lo[1] > 0 and ns and ns[1] > 0 else ('不能排除同集团解释' if ns and ns[1] <= 0 else '见数字')
json.dump(res, open(RW / 'group_eval.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False); print(json.dumps(res, ensure_ascii=False, indent=1))
