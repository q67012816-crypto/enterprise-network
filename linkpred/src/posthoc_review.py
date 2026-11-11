"""DEV_RESULTS 第 22 节：投稿前审阅提出的三项事后诊断（只用已有开发分数，不重新训练；不是事先登记的分析）。
  1 候选度数匹配：每个正例只保留与真实被投资方（other）度数同档的负例（档：0、1、2–3、4–7、8+），
    比较同一批正例上 DFGT 的 Δ（高 − 低）和低连接组的兼任增益 M1+il − M1。
  2 时间双重收紧：预测日之后取得的低连接正例（第 21.1 节）+ 按任职时间过滤的兼任特征（第 20.1 节主版本）。
  3 SEAL（官方）孤立组按角色拆分：锚点（投资方）孤立 / 只有候选方孤立。
区间：按“折 × 投资方”聚类重抽，B = 1,000，numpy.default_rng(20261103)（本研究的惯例）。
输出：out_dev_gpu10/posthoc_review.json"""
import collections, json
from pathlib import Path
import numpy as np, pandas as pd
L = Path(__file__).resolve().parents[1]; SD = L / 'out_dev_gpu10'; TD = L / 'out_dev_tenure_main'
FILES = {'M1': SD / 'scores_M1.csv', 'M1+il': SD / 'scores_M1+il.csv', 'dfgt': SD / 'scores_dfgt.csv', 'seal-official': SD / 'scores_seal-official.csv',
         'M1@tenure': TD / 'scores_M1.csv', 'M1+il@tenure': TD / 'scores_M1+il.csv'}
K = ['pair_id', 'fy', 'fold', 'anchor', 'other', 'label']
S = None
for v, f in FILES.items():
    x = pd.read_csv(f, dtype={'pair_id': str, 'anchor': str, 'other': str})[K + ['score']].rename(columns={'score': v})
    S = x if S is None else S.merge(x, on=K, validate='1:1')
deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t in S.fy.unique() for g in [pd.read_csv(L / f'data_v3x/graph_{t}.csv', dtype=str)]}
S['da'] = [deg[t][a] for t, a in zip(S.fy, S.anchor)]; S['do'] = [deg[t][o] for t, o in zip(S.fy, S.other)]
S['bin'] = pd.cut(S.do, [-1, 0, 1, 3, 7, np.inf], labels=False)
P = S[S.label == 1].set_index('pair_id'); N = S[S.label == 0].copy(); assert len(P) == 575
def auc_vs(negs):
    out = {}
    for v in FILES:
        d = negs.pair_id.map(P[v]) - negs[v]; out[v] = (d.gt(0) + .5 * d.eq(0)).groupby(negs.pair_id).mean()
    return pd.DataFrame(out)
full = auc_vs(N); matched = auc_vs(N[N.bin == N.pair_id.map(P.bin)])
W = P[['fold', 'anchor', 'da', 'do']].copy(); W['mindeg'] = W[['da', 'do']].min(axis=1); W['cluster'] = W.fold + '-' + W.anchor
W = W.join(full).join(matched.add_suffix('|m'))
W = W.join(pd.read_csv(Path('D:/claude data/scratch_label_audit/label_audit_pairs.csv'), dtype={'pair_id': str}).set_index('pair_id')['A_main'])
def boot(U, col, B=1000):
    cl = U.cluster.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(20261103); x = U[col].values
    v = [x[np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])].mean() for _ in range(B)]
    return dict(n=len(U), mean=round(float(x.mean()), 4), ci=[round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)])
def gap_boot(U, a, B=1000):
    cl = U.cluster.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(20261103)
    lo = (U.mindeg <= 1).values; x = U[a].values; f = lambda i: x[i][~lo[i]].mean() - x[i][lo[i]].mean()
    v = [f(np.concatenate([ix[c] for c in rng.choice(uc, len(uc))])) for _ in range(B)]
    return dict(est=round(float(f(np.arange(len(U)))), 4), ci=[round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)])
R = {}
Wm = W.dropna(subset=['dfgt|m']); Wm = Wm.assign(gain=Wm['M1+il|m'] - Wm['M1|m'], gain_full=Wm['M1+il'] - Wm['M1'])
R['1_degree_matched'] = dict(retained_positives=len(Wm), retained_negatives=int((N.bin == N.pair_id.map(P.bin)).sum()),
    negatives_per_positive_median=float((N.bin == N.pair_id.map(P.bin)).groupby(N.pair_id).sum().median()),
    dfgt_gap_original=gap_boot(Wm, 'dfgt'), dfgt_gap_matched=gap_boot(Wm, 'dfgt|m'), M1_gap_original=gap_boot(Wm, 'M1'), M1_gap_matched=gap_boot(Wm, 'M1|m'),
    dfgt_auc={g: dict(original=round(float(Wm.loc[m, 'dfgt'].mean()), 4), matched=round(float(Wm.loc[m, 'dfgt|m'].mean()), 4)) for g, m in (('low', Wm.mindeg <= 1), ('high', Wm.mindeg >= 2), ('iso', Wm.mindeg == 0))},
    interlock_gain_matched={g: boot(Wm[m], 'gain') for g, m in (('low', Wm.mindeg <= 1), ('iso', Wm.mindeg == 0), ('all', Wm.mindeg >= 0))},
    interlock_gain_original_same_positives={g: boot(Wm[m], 'gain_full') for g, m in (('low', Wm.mindeg <= 1), ('iso', Wm.mindeg == 0))})
W['gain_t'] = W['M1+il@tenure'] - W['M1@tenure']; W['gain'] = W['M1+il'] - W['M1']
R['2_after_cutoff_and_tenure'] = {g: dict(tenure_features=boot(W[m & (W.A_main == 'after')], 'gain_t'), original_features=boot(W[m & (W.A_main == 'after')], 'gain'))
                                  for g, m in (('low', W.mindeg <= 1), ('iso', W.mindeg == 0), ('all', W.mindeg >= 0))}
R['3_seal_isolated_roles'] = {k: boot(W[m], 'seal-official') for k, m in (('either_isolated', W.mindeg == 0), ('anchor_isolated', W.da == 0), ('only_other_isolated', (W.da > 0) & (W.do == 0)), ('both_isolated', (W.da == 0) & (W.do == 0)))}
R['3_dfgt_isolated_roles'] = {k: boot(W[m], 'dfgt') for k, m in (('either_isolated', W.mindeg == 0), ('anchor_isolated', W.da == 0), ('only_other_isolated', (W.da > 0) & (W.do == 0)))}
R['3_M1_isolated_roles'] = {k: boot(W[m], 'M1') for k, m in (('either_isolated', W.mindeg == 0), ('anchor_isolated', W.da == 0), ('only_other_isolated', (W.da > 0) & (W.do == 0)))}
json.dump(R, open(SD / 'posthoc_review.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(R, ensure_ascii=False, indent=1))
