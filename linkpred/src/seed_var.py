"""种子方差分析（开发阶段，只读验证分数）：读取 out_dev_gpu10/ 中每个 GNN 变体的逐种子分数 score_s1..score_s10，
计算每个种子单独的组内 AUC（全部 / 低连接 ≤1 / 高连接 ≥2 / 孤立 0，三折合并），报告各变体跨种子的均值和标准差；
对指定的比较，按同一种子配对求差，报告差值的均值、标准差和 10 个种子中差值 > 0 的个数。
注意：这里只反映训练随机性；第 2 节的 bootstrap 区间只反映样本抽样，两者互补。
用法：python src/seed_var.py [--out out_dev_gpu10]"""
import argparse, collections, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import rd, within_metrics

L = Path(__file__).resolve().parents[1]; D = L / 'data_v3x'
COMPS = [('dcm', 'dfgt+port+fine+il'), ('dcm', 'dcm-noA'), ('dcm', 'dcm-noB'), ('dcm', 'dcm-noC'), ('dcm-noA', 'dfgt+port+fine+il'), ('dcm-noB', 'dfgt+port+fine+il'), ('dcm-noC', 'dfgt+port+fine+il'), ('dfgt+port+fine+il+car', 'dfgt+port+fine+il'), ('dfgt+port+fine+il+ms', 'dfgt+port+fine+il'), ('dfgt+port+fine+il+hs', 'dfgt+port+fine+il'), ('dfgt+pf+il+hs', 'dfgt+pf+il'), ('dfgt+port+fine+il', 'dfgt+port+fine'), ('dfgt+port+fine', 'dfgt'), ('dfgt+il', 'dfgt'), ('dfgt+pf', 'dfgt'), ('dfgt+pf+il', 'dfgt'), ('dfgt+pf+il', 'dfgt+pf'), ('dfgt+ile', 'dfgt'), ('dfgt+ile_in', 'dfgt'),
         ('dfgt+ile+pf+il', 'dfgt'), ('dfgt-nograph+il', 'dfgt-nograph'), ('sage+ile', 'sage'), ('dfgt', 'dfgt-nograph'), ('dfgt', 'sage')]

def main(out):
    O = L / out; deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t in range(2019, 2025) for g in [rd(D / f'graph_{t}.csv')]}
    res = {}
    for f in sorted(O.glob('scores_*.csv')):
        v = f.stem[len('scores_'):]; S = pd.read_csv(f, dtype={'anchor': str, 'other': str, 'pair_id': str})
        cols = [c for c in S.columns if c.startswith('score_s')]
        if not cols: continue
        S['pos_group'] = ''; per = {}
        for c in cols:
            W = within_metrics(S, c); pos = S[S.label == 1].set_index('pair_id')
            md = np.array([min(deg[t][a], deg[t][o]) for t, a, o in zip(pos.loc[W.index].fy, W.anchor, pos.loc[W.index].other)])
            per[c] = {'all': W.AUC.mean(), 'low': W.AUC[md <= 1].mean(), 'high': W.AUC[md >= 2].mean(), 'iso': W.AUC[md == 0].mean()}
        res[v] = pd.DataFrame(per).T
    print('跨种子：均值（标准差），每个种子单独评估')
    for v, T in res.items():
        print(f'{v:18s}', '  '.join(f'{g}: {T[g].mean():.3f} ({T[g].std():.3f})' for g in ('all', 'low', 'high', 'iso')))
    print('\n按种子配对的差值：均值 ± 标准差 [差值 > 0 的种子数 / 总数]')
    out_c = {}
    for a, b in COMPS:
        if a not in res or b not in res: continue
        d = res[a] - res[b]; out_c[f'{a} − {b}'] = {g: [round(d[g].mean(), 4), round(d[g].std(), 4), int((d[g] > 0).sum())] for g in d}
        print(f'{a + " − " + b:30s}', '  '.join(f'{g}: {d[g].mean():+.3f} ± {d[g].std():.3f} [{(d[g] > 0).sum()}/{len(d)}]' for g in ('all', 'low', 'high', 'iso')))
    json.dump({'per_variant': {v: T.round(4).to_dict('index') for v, T in res.items()}, 'paired': out_c},
              open(O / 'seed_var.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='out_dev_gpu10'); main(ap.parse_args().out)
