"""功效模拟（开发阶段，只读三个验证年的分数；不碰测试年）。
问题：如果真实效应等于开发阶段的观测值，单一测试年（样本量取 2024→25 的规模）上区间不含 0 的概率有多大？
做法：
  1. 每个正例一行，取两个变体的组内 AUC 之差（10 个种子平均分数，out_dev_gpu10/）。聚类单位 = 折 × 投资方（同 eval）。
  2. 对每个组 g（全部 / 低连接 ≤1 / 孤立 0）：只用 g 中的正例；每次模拟从聚类中有放回地抽，直到 g 的正例数达到目标
     （全部 285、低连接 140、孤立 78，即 2024→25 的规模），得到一个"模拟测试年"。
  3. 每个模拟测试年内部再做 1000 次聚类 bootstrap（多项分布权重），得到该年的区间。
  4. 重复 1000 次，报告：单侧 95%（5% 分位 > 0）和双侧 95%（2.5% 分位 > 0）的比例，以及模拟测试年点估计的中位数和 5%–95% 分位。
另：C1（DFGT 的 Δ > 0）按"全部"抽样，在同一模拟年内计算 高连接 − 低连接。
注意：以开发阶段的观测效应作为真实效应，通常偏乐观（选择效应）；且三个验证年的组成（低连接 65%、孤立 43%）与测试年不同，
所以分组单独抽样。结果只作规划参考。
用法：python src/power_sim.py [--comps all|item4|all+item4] [--years 1|2] [--out out_dev_gpu10]
--years 2：模拟两个测试年合并（FY2025→26 + FY2026→27），目标样本量加倍；结果另存 power_sim_2y.json。"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import rd, within_metrics

L = Path(__file__).resolve().parents[1]; D = L / 'data_v3x'
TARGET = {'all': 285, 'low': 140, 'iso': 78}
COMPS = {   # 名称: (a, b, 组)
    'C2 dfgt − M1': ('dfgt', 'M1', ('all', 'low')),
    'C3 dfgt+pf+il − dfgt+pf': ('dfgt+pf+il', 'dfgt+pf', ('iso', 'low', 'all')),
    'C3′ dfgt+port+fine+il − dfgt+port+fine（不含结构信息）': ('dfgt+port+fine+il', 'dfgt+port+fine', ('iso', 'low', 'all')),
    'C4 dfgt+ile+pf+il − dfgt': ('dfgt+ile+pf+il', 'dfgt', ('all', 'low')),
    'S2 M1+il − M1': ('M1+il', 'M1', ('iso', 'low')),
}
ITEM4 = {   # 第 4 项（最大股东）：在已有兼任的前提下比较“加入 − 不加入”，与已定的主检验 P1 同一框架
    'C5-LR M1+il+hs − M1+il': ('M1+il+hs', 'M1+il', ('low', 'iso', 'all')),
    'C5-LR0 M1+hs − M1（不含兼任）': ('M1+hs', 'M1', ('low', 'iso', 'all')),
    "C5′ dfgt+port+fine+il+hs − dfgt+port+fine+il": ('dfgt+port+fine+il+hs', 'dfgt+port+fine+il', ('low', 'iso', 'all')),
}

def per_positive(O, variants):
    """读取各变体分数，返回每个正例一行的表：各变体 AUC、较小端度数、聚类编号。"""
    deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t in range(2019, 2025) for g in [rd(D / f'graph_{t}.csv')]}
    W = None
    for v in variants:
        S = pd.read_csv(O / ('scores_' + v.replace(':', '_').replace('|', '-') + '.csv'), dtype={'anchor': str, 'other': str, 'pair_id': str})
        S['pos_group'] = ''; w = within_metrics(S, 'score')[['AUC']].rename(columns={'AUC': v})
        if W is None:
            pos = S[S.label == 1].set_index('pair_id'); W = w.join(pos[['anchor', 'fold', 'fy', 'other']])
            W['mindeg'] = [min(deg[t][a], deg[t][o]) for t, a, o in zip(W.fy, W.anchor, W.other)]; W['cluster'] = W.fold + '-' + W.anchor
        else: W = W.join(w)
    return W.reset_index()

def sim_one(d, cl, target, rng, B=1000, S=1000):
    """d：每个正例的差值；cl：聚类编号。返回 (单侧比例, 双侧比例, 点估计分位)。"""
    uc, inv = np.unique(cl, return_inverse=True); s = np.bincount(inv, d); n = np.bincount(inv).astype(float); K = len(uc)
    lo1, lo2, est = [], [], []
    for _ in range(S):
        draw = []; tot = 0
        while tot < target:   # 有放回地抽聚类，直到正例数达到目标
            k = rng.integers(K); draw.append(k); tot += n[k]
        draw = np.array(draw); sd, nd = s[draw], n[draw]; est.append(sd.sum() / nd.sum())
        w = rng.multinomial(len(draw), np.full(len(draw), 1 / len(draw)), size=B)   # 模拟年内部的聚类 bootstrap
        b = (w @ sd) / (w @ nd)
        lo1.append(np.percentile(b, 5)); lo2.append(np.percentile(b, 2.5))
    est = np.array(est)
    return float(np.mean(np.array(lo1) > 0)), float(np.mean(np.array(lo2) > 0)), [round(float(x), 4) for x in np.percentile(est, [5, 50, 95])]

def sim_gap(W, v, rng, B=1000, S=1000):
    """C1：按"全部"抽样到 285 个正例，在模拟年内计算 Δ = 高连接 − 低连接 的区间。"""
    uc, inv = np.unique(W.cluster.values, return_inverse=True); K = len(uc); a = W[v].values
    hi, lo = (W.mindeg >= 2).values, (W.mindeg <= 1).values
    sh = np.bincount(inv, a * hi, minlength=K); sl = np.bincount(inv, a * lo, minlength=K)
    nh = np.bincount(inv, hi.astype(float), minlength=K); nl = np.bincount(inv, lo.astype(float), minlength=K); n = nh + nl
    r1, r2 = [], []
    for _ in range(S):
        draw = []; tot = 0
        while tot < TARGET['all']:
            k = rng.integers(K); draw.append(k); tot += n[k]
        draw = np.array(draw); w = rng.multinomial(len(draw), np.full(len(draw), 1 / len(draw)), size=B)
        g = (w @ sh[draw]) / (w @ nh[draw]) - (w @ sl[draw]) / (w @ nl[draw])
        r1.append(np.nanpercentile(g, 5) > 0); r2.append(np.nanpercentile(g, 2.5) > 0)
    return float(np.mean(r1)), float(np.mean(r2))

def main(out, which, years=1):
    global TARGET
    TARGET = {g: n * years for g, n in TARGET.items()}
    O = L / out; comps = {**COMPS, **ITEM4} if which == 'all+item4' else ITEM4 if which == 'item4' else COMPS
    vs = sorted({x for a, b, _ in comps.values() for x in (a, b)} | {'dfgt'})
    vs = [v for v in vs if (O / ('scores_' + v + '.csv')).exists()]
    W = per_positive(O, vs); rng = np.random.default_rng(20261103); res = {}
    grp = {'all': W.index == W.index, 'low': (W.mindeg <= 1).values, 'iso': (W.mindeg == 0).values}
    if which != 'item4':
        r = sim_gap(W, 'dfgt', rng); res['C1 Δ(dfgt) > 0'] = {'all': dict(one_sided=r[0], two_sided=r[1], n=TARGET['all'])}
        print(f"C1 Δ(dfgt) > 0  (n={TARGET['all']})  单侧 {r[0]:.3f}  双侧 {r[1]:.3f}")
    for name, (a, b, gs) in comps.items():
        if a not in W or b not in W: print('skip (scores missing):', name); continue
        res[name] = {}
        for g in gs:
            m = grp[g]; d = (W[a] - W[b]).values[m]; one, two, q = sim_one(d, W.cluster.values[m], TARGET[g], rng)
            res[name][g] = dict(dev_effect=round(float(d.mean()), 4), n=TARGET[g], one_sided=one, two_sided=two, est_q05_50_95=q)
            print(f'{name:58s} {g:4s} (n={TARGET[g]})  开发效应 {d.mean():+.3f}  单侧 {one:.3f}  双侧 {two:.3f}  模拟点估计 5/50/95% {q}', flush=True)
    f = O / ('power_sim.json' if years == 1 else f'power_sim_{years}y.json'); old = json.load(open(f, encoding='utf-8')) if f.exists() else {}
    old.update(res); json.dump(old, open(f, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='out_dev_gpu10'); ap.add_argument('--comps', default='all', choices=['all', 'item4', 'all+item4']); ap.add_argument('--years', type=int, default=1)
    a = ap.parse_args(); main(a.out, a.comps, a.years)
