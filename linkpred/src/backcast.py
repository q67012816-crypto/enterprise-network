"""DEV_RESULTS 第 26.5 节：2015–2018 年回测（构建、特征、只评分一次、评估）。冻结的 build_v3.py / interlock_feats.py / paper_dev.py 都不改：
build 步骤读取 build_v3.py 的源码，只做下面 REPL 列出的年份常量和输入输出路径替换（每处必须恰好出现一次）后在本进程执行。
用法（按顺序）：
  python src/backcast.py combine    # 合并新下载的回测原始数据与现有原始数据 → data/interim/backcast/combined/
  python src/backcast.py regress    # 不做任何替换执行 build_v3.py（输出到临时目录），须与 data_v3x/pairs.csv 逐字节相同
  python src/backcast.py build      # 按 REPL 替换后执行 → linkpred/data_v3b/
  python src/backcast.py feats      # 兼任特征（冻结的 interlock_feats，年份 2015–2018）→ linkpred/out_backcast/feat/
  python src/backcast.py check      # 评分前允许查看的覆盖统计（不看标签与分数的关系）
  python src/backcast.py score      # 只评分一次：M1、M1+il，扩展窗口三个测试年（2016、2017、2018）
  python src/backcast.py eval       # 主检验与报告"""
import collections, contextlib, hashlib, io, json, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
L = Path(__file__).resolve().parents[1]; P0 = L.parent; RAW = P0 / 'data/interim'; BC = RAW / 'backcast'; CB = BC / 'combined'
DB = L / 'data_v3b'; OB = L / 'out_backcast'; FT = OB / 'feat'; SRC = Path(__file__).resolve().parent
cmd = sys.argv[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rd = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
REPL = [   # (原文, 替换) —— 只改年份常量和输入输出路径
    ("FY = list(range(ARGS.first_year, 2025))", "FY = list(range(2015, 2019))"),
    ("V = read(P / 'data/interim/invest_panel_raw.csv'); U = read(P / 'data/processed/universe_firms.csv'); O = read(P / 'data/interim/company_overview.csv'); A = read(P / 'data/interim/filings_A001.csv')",
     "V = read(P / 'data/interim/backcast/combined/invest_panel_raw.csv'); U = read(P / 'data/interim/backcast/combined/universe_firms.csv'); O = read(P / 'data/interim/backcast/combined/company_overview.csv'); A = read(P / 'data/interim/backcast/combined/filings_A001.csv')"),
    ("for t in range(2019, 2026):\n    c = C(t)", "for t in range(2014, 2020):\n    c = C(t)"),
    ("F = read(P / 'data/interim/fin_multi_raw.csv')", "F = read(P / 'data/interim/backcast/combined/fin_multi_raw.csv')"),
    ("fp = pd.read_csv(P / 'reruns/repaired-edges/fin_panel_rebuilt.csv'", "fp = pd.read_csv(P / 'data/interim/backcast/combined/fin_panel_rebuilt.csv'"),
    ("split = {2019: 'dev_early', 2020: 'dev_early', 2021: 'train', 2022: 'train', 2023: 'validation', 2024: 'test'}", "split = {2015: 'backcast', 2016: 'backcast', 2017: 'backcast', 2018: 'backcast'}"),
    ("for t in range(2020, 2026):\n    Gu = und(G[t])", "for t in range(2015, 2020):\n    Gu = und(G[t])"),
    ("for t in range(2019, 2026):\n    pd.DataFrame([dict(src=a_", "for t in range(2014, 2020):\n    pd.DataFrame([dict(src=a_"),
]
TEST = {2016: (2015,), 2017: (2015, 2016), 2018: (2015, 2016, 2017)}

def run_build(out, repl):
    s = (SRC / 'build_v3.py').read_text(encoding='utf-8')
    for a, b in repl:
        assert s.count(a) == 1, f'替换原文必须恰好出现一次：{a[:60]}'; s = s.replace(a, b)
    argv0 = sys.argv; sys.argv = ['build_v3.py', '--out', str(out), '--first-year', '2019']
    ns = dict(__name__='__main__', __file__=str(SRC / 'build_v3.py'))
    with contextlib.redirect_stdout(io.StringIO()) as buf: exec(compile(s, str(SRC / 'build_v3.py'), 'exec'), ns)
    sys.argv = argv0; return buf.getvalue()

if cmd == 'combine':
    CB.mkdir(parents=True, exist_ok=True)
    cat = lambda a, b: pd.concat([rd(a), rd(b)], ignore_index=True)
    f = cat(RAW / 'filings_A001.csv', BC / 'filings_A001_2015_2018.csv').drop_duplicates('rcept_no'); f.to_csv(CB / 'filings_A001.csv', index=False)
    cat(RAW / 'company_overview.csv', BC / 'company_overview_new.csv').drop_duplicates('corp_code').to_csv(CB / 'company_overview.csv', index=False)
    U0 = rd(P0 / 'data/processed/universe_firms.csv'); U1 = rd(BC / 'universe_backcast.csv')
    pd.DataFrame({'corp_code': sorted(set(U0.corp_code) | set(U1.corp_code))}).to_csv(CB / 'universe_firms.csv', index=False)
    V0 = rd(RAW / 'invest_panel_raw.csv'); V1 = rd(BC / 'invest_panel_raw_2014_2018.csv')
    pd.concat([V1, V0[V0.bsns_year == '2019']], ignore_index=True).to_csv(CB / 'invest_panel_raw.csv', index=False)
    F0 = rd(RAW / 'fin_multi_raw.csv'); F1 = rd(BC / 'fin_multi_raw_2015_2017.csv'); pd.concat([F1, F0], ignore_index=True).to_csv(CB / 'fin_multi_raw.csv', index=False)
    # 财务面板：用 reruns/repaired-edges/src/fin_panel.py 的同一函数，在新下载的 2015–2017 行上构建，再与现有面板合并
    sys.path.insert(0, str(P0 / 'reruns/repaired-edges/src')); import fin_panel as FPm
    orig = FPm.pd.read_csv
    FPm.pd.read_csv = lambda p, *a, **k: orig(BC / 'fin_multi_raw_2015_2017.csv' if str(p).endswith('fin_multi_raw.csv') else p, *a, **k)
    new = FPm.build(); FPm.pd.read_csv = orig
    old = pd.read_csv(P0 / 'reruns/repaired-edges/fin_panel_rebuilt.csv', dtype={'corp_code': str, 'currencies': str})
    pd.concat([new[new.fy.between(2015, 2017)], old], ignore_index=True).to_csv(CB / 'fin_panel_rebuilt.csv', index=False)
    print('combined:', {p.name: sum(1 for _ in open(p, encoding='utf-8')) - 1 for p in CB.glob('*.csv')})
elif cmd == 'regress':
    tmp = P0.parent / 'scratch_backcast_regress'; run_build(tmp, [])
    print('regress identical to data_v3x:', sha(tmp / 'pairs.csv') == sha(L / 'data_v3x' / 'pairs.csv'))
elif cmd == 'build':
    if (DB / 'pairs.csv').exists(): sys.exit('data_v3b 已存在，拒绝覆盖')
    log = run_build(DB, REPL); print(log[-3000:])
elif cmd == 'feats':
    import interlock_feats as IF
    if (FT / 'interlock_feats.csv').exists(): sys.exit('特征已存在，拒绝覆盖')
    FT.mkdir(parents=True, exist_ok=True); IF.D = DB; IF.O = FT; IF.YEARS = range(2015, 2019)
    orig = IF.pd.read_csv
    IF.pd.read_csv = lambda p, *a, **k: orig(BC / 'exec_panel_raw_2015_2018.csv' if str(p).endswith('exec_panel_raw.csv') else p, *a, **k)
    try:
        with contextlib.redirect_stdout(io.StringIO()): IF.main()
    finally: IF.pd.read_csv = orig
    X = pd.read_csv(DB / 'pairs.csv', usecols=['fy']); X[X.fy.between(2015, 2018)].iloc[:, :0].to_csv(FT / 'pair_feats.csv', index_label='row')   # M1/M1+il 不用点对特征，只提供行号
    print('feats written')
elif cmd == 'check':
    X = pd.read_csv(DB / 'pairs.csv', dtype={'anchor': str, 'other': str}); F = pd.read_csv(FT / 'interlock_feats.csv', index_col='row')
    G = {t: pd.read_csv(DB / f'graph_{t}.csv', dtype=str) for t in range(2014, 2020)}; el = pd.read_csv(DB / 'eligible.csv', dtype=str)
    pos = X[X.label == 1]; deg = {t: collections.Counter(list(G[t].src) + list(G[t].dst)) for t in range(2015, 2019)}
    md = pd.Series([min(deg[t][a], deg[t][o]) for t, a, o in zip(pos.fy, pos.anchor, pos.other)], index=pos.index)
    cov = ((F.il_has_a == 1) & (F.il_has_o == 1)).groupby(X.loc[F.index].fy).mean()
    out = dict(eligible=el.groupby('fy').size().to_dict(), edges={t: len(g) for t, g in G.items()}, positives=pos.groupby('fy').size().to_dict(),
               positives_low=pos[md <= 1].groupby('fy').size().to_dict(), positives_iso=pos[md == 0].groupby('fy').size().to_dict(),
               interlock_coverage_both=cov.round(4).to_dict(), sample_flow=pd.read_csv(DB / 'sample_flow.csv', index_col=0).to_dict())
    json.dump(out, open(OB / 'precheck.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=str); print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
elif cmd == 'score':
    if any(OB.glob('scores_*.csv')): sys.exit('已有分数文件，拒绝运行（只评分一次）')
    argv0 = sys.argv; sys.argv = ['paper_dev.py', 'none', '--out', 'out_backcast', '--device', 'cpu']
    import paper_dev as PD
    sys.argv = argv0; PD.D = DB; PD.FE = FT; PD.GRAPH_YEARS = range(2015, 2019); PD.PF_ZERO_FROM = 2100
    X, N, G = PD.load(); cols = ['pair_id', 'fy', 'anchor', 'other', 'label']; man = dict(started=time.strftime('%Y-%m-%d %H:%M:%S %z'), files={})
    for v in ('M1', 'M1+il'):
        model, opt = PD.VARIANTS[v]; rows = []
        for te, tr in TEST.items():
            Ys = PD.node_tables(N, G, list(tr) + [te], tr, False); s = PD.run_lr(X, Ys, tr, te, opt)
            rows.append(X[X.fy == te][cols].assign(fold=f'B{te}', score=s))
        f = OB / PD.sfile(v).name; pd.concat(rows).to_csv(f, index=False); man['files'][f.name] = sha(f)
    man['finished'] = time.strftime('%Y-%m-%d %H:%M:%S %z'); man['data_sha256'] = {p.name: sha(p) for p in sorted(DB.glob('*.csv'))}; man['feat_sha256'] = {p.name: sha(p) for p in sorted(FT.glob('*.csv'))}
    json.dump(man, open(OB / 'score_manifest.json', 'w', encoding='utf-8'), indent=1); print('scored once:', man['files'])
elif cmd == 'eval':
    deg = {t: collections.Counter(list(g.src) + list(g.dst)) for t in range(2016, 2019) for g in [pd.read_csv(DB / f'graph_{t}.csv', dtype=str)]}
    def per_pos(f):
        S = pd.read_csv(f, dtype={'pair_id': str, 'anchor': str, 'other': str}); r = {}
        for pid, g in S.groupby('pair_id'):
            p = g[g.label == 1].iloc[0]; n = g.score.values[g.label.values == 0]; s = p.score; rk = 1 + (n > s).sum() + .5 * (n == s).sum()
            r[pid] = (p.fold, p.anchor, int(p.fy), min(deg[p.fy][p.anchor], deg[p.fy][p.other]), ((n < s).sum() + .5 * (n == s).sum()) / len(n), rk)
        return pd.DataFrame(r, index=['fold', 'anchor', 'fy', 'md', 'auc', 'rank']).T
    A_, B_ = per_pos(OB / 'scores_M1.csv'), per_pos(OB / 'scores_M1+il.csv'); W = A_[['fold', 'anchor', 'fy', 'md']].copy()
    W['md'] = W.md.astype(int); W['fy'] = W.fy.astype(int); W['cl'] = W.fold + '-' + W.anchor
    for k, P in (('M1', A_), ('il', B_)):
        r = P['rank'].reindex(W.index).astype(float); W[f'{k}|auc'] = P.auc.reindex(W.index).astype(float); W[f'{k}|H1'] = (r <= 1).astype(float); W[f'{k}|H10'] = (r <= 10).astype(float); W[f'{k}|rr'] = 1 / r
    def boot(U, d, m, B, seed=20261103):
        cl = U.cl.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(seed); v = []
        for _ in range(B):
            i = np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]); mm = m[i]
            if mm.any(): v.append(d[i][mm].mean())
        return [round(float(d[m].mean()), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4), round(float(np.percentile(v, 5)), 4), int(m.sum())]
    GR = lambda U: {'low': (U.md <= 1).values, 'iso': (U.md == 0).values, 'all': np.ones(len(U), bool)}
    res = dict(pooled={g: boot(W, (W['il|auc'] - W['M1|auc']).values, m, 10000) for g, m in GR(W).items()},
               per_year={t: {g: boot(W[W.fy == t], (W[W.fy == t]['il|auc'] - W[W.fy == t]['M1|auc']).values, m, 1000) for g, m in GR(W[W.fy == t]).items()} for t in TEST},
               ranking={g: {mm: boot(W, (W[f'il|{mm}'] - W[f'M1|{mm}']).values, m, 1000) for mm in ('H1', 'H10', 'rr')} for g, m in GR(W).items()},
               levels={g: {k: round(float(W.loc[m, f'{k}|auc'].mean()), 4) for k in ('M1', 'il')} for g, m in GR(W).items()})
    lo = res['pooled']['low']; res['verdict'] = '回测复现' if lo[3] > 0 else ('方向一致但不显著' if lo[0] > 0 else '未复现')
    json.dump(res, open(OB / 'backcast_eval.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False); print(json.dumps(res, ensure_ascii=False, indent=1))
