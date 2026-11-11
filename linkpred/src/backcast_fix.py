"""DEV_RESULTS 第 26.8 节：回测纠错运行。唯一改动：历史任职特征 il_hist 的起点从写死的 2019 改为 2015（回测高管数据的第一年）。
冻结的 interlock_feats.py 和已有的 backcast.py 都不改：两者都读取源码，做逐字替换（断言恰好出现一次）后在本进程执行。
输出：linkpred/out_backcast_fix/（原运行 out_backcast/ 原样保留）。
用法：python src/backcast_fix.py feats | verify | score | eval"""
import contextlib, io, shutil, sys
from pathlib import Path
import pandas as pd
SRC = Path(__file__).resolve().parent; L = SRC.parents[0]; P0 = L.parent
DB = L / 'data_v3b'; OB = L / 'out_backcast_fix'; FT = OB / 'feat'; OLD = L / 'out_backcast' / 'feat'; BC = P0 / 'data/interim/backcast'
cmd = sys.argv[1]

def exec_patched(path, repl, argv, extra=None):
    s = path.read_text(encoding='utf-8')
    for a, b in repl:
        assert s.count(a) == 1, f'替换原文必须恰好出现一次：{a}'; s = s.replace(a, b)
    ns = dict(__name__='__patched__', __file__=str(path)); ns.update(extra or {})
    argv0 = sys.argv; sys.argv = argv
    try: exec(compile(s, str(path), 'exec'), ns)
    finally: sys.argv = argv0
    return ns

if cmd == 'feats':
    if (FT / 'interlock_feats.csv').exists(): sys.exit('特征已存在，拒绝覆盖')
    FT.mkdir(parents=True, exist_ok=True)
    ns = exec_patched(SRC / 'interlock_feats.py', [('for s in range(2019, t + 1):', 'for s in range(2015, t + 1):')], ['interlock_feats.py'])
    ns['D'] = DB; ns['O'] = FT; ns['YEARS'] = range(2015, 2019)
    orig = pd.read_csv
    pd.read_csv = lambda p, *a, **k: orig(BC / 'exec_panel_raw_2015_2018.csv' if str(p).endswith('exec_panel_raw.csv') else p, *a, **k)
    try:
        with contextlib.redirect_stdout(io.StringIO()): ns['main']()
    finally: pd.read_csv = orig
    shutil.copyfile(OLD / 'pair_feats.csv', FT / 'pair_feats.csv'); print('feats written')
elif cmd == 'verify':
    N = pd.read_csv(FT / 'interlock_feats.csv', index_col='row'); O_ = pd.read_csv(OLD / 'interlock_feats.csv', index_col='row')
    X = pd.read_csv(DB / 'pairs.csv', usecols=['fy', 'label'])
    assert (N.index == O_.index).all()
    same = {c: bool((N[c] == O_[c]).all()) for c in N.columns if c != 'il_hist'}
    nz = (N.il_hist > 0).groupby(X.loc[N.index].fy).sum().to_dict()
    print('other five columns identical:', same); print('nonzero il_hist rows by fy:', nz)
elif cmd in ('score', 'eval'):
    exec_patched(SRC / 'backcast.py', [("OB = L / 'out_backcast';", "OB = L / 'out_backcast_fix';")], ['backcast.py', cmd])
