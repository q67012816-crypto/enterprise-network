"""修订 A8.1（只报告，不判定）：确认运行时兼任信号的时间泄漏敏感性分析。规则与 DEV_RESULTS 第 20.1 节相同（tenure_rules.py）。
不修改任何 A8 冻结的文件；只读取冻结运行的输出（out_confirm），不写入它。必须在 paper_confirm.py 正式运行完成之后运行。

正式运行（2027 年，paper_confirm.py 完成后；--data / --feat 与冻结运行相同）：
  python src/confirm_tenure.py feats --mode main   --data data_v3x_2027 --feat <冻结运行的特征目录>
  python src/confirm_tenure.py run   --mode main   --data data_v3x_2027 --ref out_confirm
  python src/confirm_tenure.py eval  --mode main   --data data_v3x_2027
  （严格版本：以上三条把 --mode main 换成 --mode strict）
演练（只用开发年，与 paper_confirm.py --dry-run 相同的年份；不碰 2024→25）：
  python src/confirm_tenure.py feats --mode main --dry-run
  python src/confirm_tenure.py run   --mode main --dry-run --ref out_confirm_dryrun
  python src/confirm_tenure.py eval  --mode main --dry-run

输出目录：linkpred/out_confirm_tenure_{mode}（演练：out_confirm_tenure_{mode}_dryrun），其下 feat/ 为特征。
  - feats：按第 20.1 节的规则排除可能在 t 年 12 月 31 日之后上任的行（主版本保留无法解析的行，严格版本也排除），
           再用冻结的 interlock_feats.main() 重算兼任特征（年份 2019 … 测试年）；pair_feats.csv、hyslr_feats.csv 从冻结特征目录复制。
  - run：只跑 M1+il 和 dfgt+port+fine+il（与 paper_confirm.py 相同的年份、种子 1–10、早停和打分流程），先写验证分数再写测试分数并记录 SHA-256；
         再从冻结运行的输出目录复制 scores_M1.csv、scores_dfgt+port+fine.csv（先按其 manifest.json 核对 SHA-256），作为比较基准。
  - eval：用 paper_dev.evaluate()，B = 10,000，种子 20261103（与 P1 同一套抽样），比较 M1+il − M1 和 dfgt+port+fine+il − dfgt+port+fine。"""
import argparse, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

CONFIRM = dict(train=(2019, 2020, 2021, 2022, 2023), val=2024, test=2025)   # 与 paper_confirm.py 相同
DRY = dict(train=(2019, 2020, 2021), val=2022, test=2023)
SEEDS = list(range(1, 11))
MODELS = ['M1+il', 'dfgt+port+fine+il']
BASE = ['M1', 'dfgt+port+fine']
PAIRS = 'M1+il:M1,dfgt+port+fine+il:dfgt+port+fine'

ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['feats', 'run', 'eval']); ap.add_argument('--mode', choices=['main', 'strict'], required=True)
ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--data', default='data_v3x'); ap.add_argument('--feat', default='out_dev', help='冻结运行的特征目录（相对 linkpred/）')
ap.add_argument('--ref', default='', help='冻结运行的输出目录（run 时必需）'); ap.add_argument('--device', default='cuda')
A = ap.parse_args()
Y = DRY if A.dry_run else CONFIRM
if Y['test'] == 2024: sys.exit('测试年不能是 2024')
L = Path(__file__).resolve().parents[1]; P0 = L.parent; sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = L / (f'out_confirm_tenure_{A.mode}' + ('_dryrun' if A.dry_run else '')); FT = OUT / 'feat'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git_head():
    try: return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=L, capture_output=True, text=True).stdout.strip()
    except Exception: return ''
def manifest(upd):
    f = OUT / 'manifest.json'; m = json.load(open(f, encoding='utf-8')) if f.exists() else {}
    m.update(upd); json.dump(m, open(f, 'w', encoding='utf-8'), indent=1, ensure_ascii=False); return m

if A.cmd == 'feats':
    from tenure_rules import classify
    import interlock_feats as IF
    if (FT / 'interlock_feats.csv').exists(): sys.exit(f'{FT} 已有特征文件，拒绝覆盖')
    FT.mkdir(parents=True, exist_ok=True)
    IF.D = L / A.data; IF.O = FT; IF.YEARS = range(2019, Y['test'] + 1)
    src = P0 / 'data/interim/exec_panel_raw.csv'
    E = pd.read_csv(src, dtype=str, keep_default_na=False)
    fy = pd.to_numeric(E.bsns_year, errors='coerce'); rdate = E.rcept_no.str[:8]
    res = [classify(h, int(t), r) if t == t and len(r) == 8 else ('unparsed', None) for h, t, r in zip(E.hffc_pd, fy, rdate)]
    cat = pd.Series([c for c, _ in res], index=E.index); ex = [x for _, x in res]
    drop = pd.Series([bool(x) if x is not None else (A.mode == 'strict') for x in ex], index=E.index)
    ok = E.api_status.eq('000') & E.nm.ne('') & E.birth_ym.str.replace(r'\D', '', regex=True).str.len().eq(6) & rdate.le((fy + 1).fillna(0).astype(int).astype(str) + '0430') & fy.between(2019, Y['test'])
    stats = dict(rows_used=int(ok.sum()), category_share_by_fy=pd.crosstab(E.bsns_year[ok], cat[ok], normalize='index').round(4).to_dict('index'),
                 dropped_share_by_fy={k: round(float(v), 4) for k, v in drop[ok].groupby(E.bsns_year[ok]).mean().items()}, dropped_share=round(float(drop[ok].mean()), 4))
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    tmp = FT / 'exec_panel_filtered.csv'; E[~drop].to_csv(tmp, index=False)
    orig = pd.read_csv
    IF.pd.read_csv = lambda path, *a, **k: orig(tmp if str(path).endswith('exec_panel_raw.csv') else path, *a, **k)
    try: IF.main()
    finally: IF.pd.read_csv = orig; tmp.unlink()
    for f in ('pair_feats.csv', 'hyslr_feats.csv'): shutil.copyfile(L / A.feat / f, FT / f)
    manifest(dict(mode=A.mode, dry_run=A.dry_run, years=Y, feats=dict(started_command=' '.join(sys.argv), git_head=git_head(), exec_panel_sha256=sha(src),
                  tenure_rules_sha256=sha(Path(__file__).parent / 'tenure_rules.py'), interlock_feats_py_sha256=sha(Path(__file__).parent / 'interlock_feats.py'),
                  frozen_feat_dir=A.feat, output_sha256={p.name: sha(p) for p in sorted(FT.glob('*.csv'))}, **stats)))

elif A.cmd == 'run':
    if not A.ref: sys.exit('run 需要 --ref <冻结运行的输出目录>')
    REF = L / A.ref; rm = json.load(open(REF / 'manifest.json', encoding='utf-8'))
    if 'finished' not in rm: sys.exit(f'{REF} 的冻结运行尚未完成')
    if rm['years'] != {k: list(v) if isinstance(v, tuple) else v for k, v in Y.items()}: sys.exit(f'年份与冻结运行不一致：{rm["years"]}')
    if any(OUT.glob('scores_*.csv')) or any(OUT.glob('val_scores_*.csv')): sys.exit(f'{OUT} 已有分数文件，拒绝运行（只运行一次）')
    if not (FT / 'interlock_feats.csv').exists(): sys.exit('先运行 feats')
    argv0 = sys.argv; sys.argv = ['paper_dev.py', 'none', '--out', OUT.relative_to(L).as_posix(), '--seeds', str(len(SEEDS)), '--device', A.device]
    import paper_dev as PD
    sys.argv = argv0
    PD.D = L / A.data; PD.FE = FT; PD.GRAPH_YEARS = range(2019, Y['test'] + 1); PD.PF_ZERO_FROM = Y['test'] + 1
    X, N, G = PD.load(); years = list(Y['train']) + [Y['val'], Y['test']]
    for t in years:
        if not (X.fy == t).any(): sys.exit(f'数据中没有 fy = {t} 的点对')
    man = manifest(dict(run=dict(command=' '.join(sys.argv), started=time.strftime('%Y-%m-%d %H:%M:%S %z'), git_head=git_head(), ref=A.ref, ref_git_head=rm.get('git_head'),
                    data_sha256={p.name: sha(p) for p in sorted(PD.D.glob('*.csv'))}, files={}, val_auc={}, seconds={})))
    VA, TE = X[X.fy == Y['val']], X[X.fy == Y['test']]; cols = ['pair_id', 'fy', 'anchor', 'other', 'label']
    for v in MODELS:   # 与 paper_confirm.py 的循环相同
        model, opt = PD.VARIANTS[v]; t0 = time.time(); Ys = PD.node_tables(N, G, years, Y['train'], opt.get('target', False))
        if model == 'lr': sv, st = PD.run_lr(X, Ys, Y['train'], Y['val'], opt, te_year=Y['test']); per_v = per_t = {}
        else:
            res = {sd: PD.run_gnn(X, Ys, Y['train'], Y['val'], model, opt, sd, te_year=Y['test']) for sd in SEEDS}
            per_v = {f'score_s{sd}': r[0] for sd, r in res.items()}; per_t = {f'score_s{sd}': r[1] for sd, r in res.items()}
            sv, st = np.mean(list(per_v.values()), 0), np.mean(list(per_t.values()), 0)
        fv = OUT / ('val_' + PD.sfile(v).name); VA[cols].assign(fold='V', score=sv, **per_v).to_csv(fv, index=False); man['run']['files'][fv.name] = sha(fv); manifest(man)
        ft = OUT / PD.sfile(v).name; TE[cols].assign(fold='T', score=st, **per_t).to_csv(ft, index=False); man['run']['files'][ft.name] = sha(ft)
        man['run']['val_auc'][v] = round(PD.fast_auc(VA, sv), 4); man['run']['seconds'][v] = round(time.time() - t0, 1); manifest(man)
        print(v, 'val', man['run']['val_auc'][v], f"{man['run']['seconds'][v]}s", flush=True)
    for v in BASE:   # 比较基准：冻结运行的分数，核对哈希后复制
        f = REF / PD.sfile(v).name
        if sha(f) != rm['files'][f.name]: sys.exit(f'{f} 的 SHA-256 与冻结运行的 manifest.json 不一致')
        shutil.copyfile(f, OUT / f.name); man['run']['files']['ref:' + f.name] = rm['files'][f.name]
    man['run']['finished'] = time.strftime('%Y-%m-%d %H:%M:%S %z'); manifest(man); print('done')

else:
    argv0 = sys.argv; sys.argv = ['paper_dev.py', 'none', '--out', OUT.relative_to(L).as_posix(), '--boot', '10000', '--only', PAIRS, '--gaps', ',', '--save', 'tenure_eval_boot10000.json']
    import paper_dev as PD
    sys.argv = argv0
    PD.D = L / A.data; PD.FE = FT; PD.GRAPH_YEARS = range(2019, Y['test'] + 1)
    have = sorted(p.name for p in OUT.glob('scores_*.csv'))
    if have != sorted(PD.sfile(v).name for v in MODELS + BASE): sys.exit(f'{OUT} 中的分数文件不对：{have}')
    PD.evaluate(); manifest(dict(eval=dict(finished=time.strftime('%Y-%m-%d %H:%M:%S %z'), result_sha256=sha(OUT / 'tenure_eval_boot10000.json'))))
