"""DEV_RESULTS 第 20.1 节：兼任信号的时间泄漏检查（只用开发年；不修改冻结的 interlock_feats.py）。
用 hffc_pd（재직기간）推算每一行（人 × 公司 × FY t）的上任时间，排除可能在 t 年 12 月 31 日之后上任的行，再按原定义重算兼任特征。
基准日一律取年报提交日（rcept_no 前 8 位），解析规则见第 20.1 节。
用法：
  python src/interlock_tenure.py feats --mode main     # 输出 out_dev_tenure_main/interlock_feats.csv（另复制 pair_feats.csv），并打印覆盖率
  python src/interlock_tenure.py feats --mode strict   # 无法解析的行也排除
  python src/interlock_tenure.py run --mode main --variant M1+il --seeds 10 --device cuda   # 用新特征跑 paper_dev 的变体，分数写同一目录"""
import argparse, collections, re, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
L = Path(__file__).resolve().parents[1]; P0 = L.parent
ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['feats', 'run']); ap.add_argument('--mode', choices=['main', 'strict'], required=True)
ap.add_argument('--variant'); ap.add_argument('--seeds', type=int, default=10); ap.add_argument('--device', default='cpu')
A = ap.parse_args(); OUT = L / f'out_dev_tenure_{A.mode}'; OUT.mkdir(exist_ok=True)

from tenure_rules import months_between, classify   # 规则移到 tenure_rules.py（与 confirm_tenure.py 共用），内容未改

if A.cmd == 'feats':
    import interlock_feats as IF
    E = pd.read_csv(P0 / 'data/interim/exec_panel_raw.csv', dtype=str, keep_default_na=False)
    fy = pd.to_numeric(E.bsns_year, errors='coerce'); rdate = E.rcept_no.str[:8]
    res = [classify(h, int(t), r) if t == t and len(r) == 8 else ('unparsed', None) for h, t, r in zip(E.hffc_pd, fy, rdate)]
    E['cat'] = [c for c, _ in res]; ex = np.array([x for _, x in res], dtype=object)
    E['drop'] = [bool(x) if x is not None else (A.mode == 'strict') for x in ex]
    ok = E.api_status.eq('000') & E.nm.ne('') & E.birth_ym.str.replace(r'\D', '', regex=True).str.len().eq(6) & E.rcept_no.str[:8].le((fy + 1).fillna(0).astype(int).astype(str) + '0430')
    S = E[ok]
    print('rows used by interlock_feats (strict as-of):', len(S))
    print('category share by FY:'); print(pd.crosstab(S.bsns_year, S.cat, normalize='index').round(4).to_string())
    print('dropped row share by FY:', S.groupby('bsns_year')['drop'].mean().round(4).to_dict(), 'overall', round(S['drop'].mean(), 4))
    pid = S.nm.str.replace(r'\s+', '', regex=True) + '|' + S.birth_ym.str.replace(r'\D', '', regex=True)
    print('persons with every row dropped:', round(float((S.assign(pid=pid).groupby('pid')['drop'].all()).mean()), 4))
    # 按原 persons() 的读取方式，把被排除的行去掉后写一个临时 CSV，再让 interlock_feats 读它
    tmp = OUT / 'exec_panel_filtered.csv'; E[~E['drop']].drop(columns=['cat', 'drop']).to_csv(tmp, index=False)
    orig_read = pd.read_csv
    def patched(path, *a, **k):
        return orig_read(tmp if str(path).endswith('exec_panel_raw.csv') else path, *a, **k)
    IF.pd.read_csv = patched; IF.O = OUT
    IF.main(); IF.pd.read_csv = orig_read
    shutil.copyfile(L / 'out_dev' / 'pair_feats.csv', OUT / 'pair_feats.csv')
    for f in ('hyslr_feats.csv',):
        if (L / 'out_dev' / f).exists(): shutil.copyfile(L / 'out_dev' / f, OUT / f)
    a = pd.read_csv(L / 'out_dev' / 'interlock_feats.csv', index_col='row'); b = pd.read_csv(OUT / 'interlock_feats.csv', index_col='row')
    X = pd.read_csv(L / 'data_v3x' / 'pairs.csv', usecols=['fy', 'anchor', 'other', 'label']).loc[a.index]
    deg = {t: collections.Counter(list(pd.read_csv(L / 'data_v3x' / f'graph_{t}.csv', dtype=str).src) + list(pd.read_csv(L / 'data_v3x' / f'graph_{t}.csv', dtype=str).dst)) for t in range(2019, 2024)}
    X = pd.read_csv(L / 'data_v3x' / 'pairs.csv', usecols=['fy', 'anchor', 'other', 'label'], dtype={'anchor': str, 'other': str}).loc[a.index]
    md = np.array([min(deg[t][u], deg[t][v]) for t, u, v in zip(X.fy, X.anchor, X.other)])
    chg = (a[['il_direct', 'il_inside', 'il_2hop', 'il_hist']] != b[['il_direct', 'il_inside', 'il_2hop', 'il_hist']]).any(axis=1)
    sig_a = (a[['il_direct', 'il_inside', 'il_2hop', 'il_hist']] > 0).any(axis=1); sig_b = (b[['il_direct', 'il_inside', 'il_2hop', 'il_hist']] > 0).any(axis=1)
    for name, m in (('all pairs', np.ones(len(X), bool)), ('positives', X.label.values == 1), ('low positives', (X.label.values == 1) & (md <= 1)), ('isolated positives', (X.label.values == 1) & (md == 0))):
        print(f'{name:20s} n {m.sum():6d}  features changed {chg[m].mean():.4f}  any signal before {sig_a[m].mean():.4f} after {sig_b[m].mean():.4f}')
    tmp.unlink()
else:
    argv0 = sys.argv; sys.argv = ['paper_dev.py', 'none', '--out', OUT.name, '--seeds', str(A.seeds), '--device', A.device]
    import paper_dev as PD
    sys.argv = argv0; PD.FE = OUT; PD.run(A.variant)
