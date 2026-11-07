"""确认性运行（修订 A8 第 4、5、9 节）：在训练年训练、验证年早停，测试年只打分一次。
  正式运行（2027 年，FY2025→26 数据收集并通过第 2.1 节的门槛之后）：
      python src/paper_confirm.py run --data data_v3x_2027 --out out_confirm
    训练 2019→20 … 2023→24，验证（只用于早停）2024→25，测试 2025→26。
  演练（只用开发年，不碰 2024→25）：
      python src/paper_confirm.py run --dry-run --out out_confirm_dryrun
    训练 2019→20 … 2021→22，验证 2022→23，"测试" 2023→24。演练只用来检查流程，结果不进入任何检验或报告。
冻结后只允许修改 CONFIRM 和 DRY 中的年份、--data / --feat 路径，以及 paper_dev.py 的 GRAPH_YEARS（评估用）；其余代码不变（第 5.3 节）。

规则（第 5.2、9 节）：
  - 输出目录里已有任何分数文件或 manifest.json 时，拒绝运行（只运行一次）；
  - 正式模式下测试年必须是 2025，且数据中必须有该年的点对；任何模式下测试年都不能是 2024（2024→25 只作验证年）；
  - 每个 GNN 变体跑 10 个种子（1–10），测试分数 = 10 个种子的平均，另存每个种子的分数；逻辑回归只拟合一次（确定性）；
  - GNN 在验证组内 AUC 最好的一轮保存参数，训练结束后用这一组参数给测试年打分一次；
  - 每个模型先写验证分数文件并记录 SHA256，再写测试分数文件并记录 SHA256；同时记录代码提交、源文件和数据文件的 SHA256（manifest.json）；
  - 计算第 2.4 节的覆盖率（兼任、最大股东），写入 manifest.json；低于门槛时照常打分，但依赖它的检验按第 2.4 节报告为“不可解读”。
分数文件格式与 paper_dev.py 相同（fold = 'T'），评估直接用 paper_dev.py（年份常量 GRAPH_YEARS 须包含测试年）：
      python src/paper_dev.py eval --out <out> --boot 10000 --only "M1+il:M1,dfgt:M1" --gaps dfgt --save confirm_eval_boot10000.json
      python src/paper_dev.py eval --out <out> --boot 1000 --save confirm_eval_boot1000.json"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

CONFIRM = dict(train=(2019, 2020, 2021, 2022, 2023), val=2024, test=2025)
DRY = dict(train=(2019, 2020, 2021), val=2022, test=2023)
SEEDS = list(range(1, 11))
# 第 5.3 节的模型（M_a … M_h、M_f′、M_g′、M_x1–M_x3），加上第 7.2 节报告项和敏感性分析需要的变体
MODELS = ['M1', 'M1+il', 'sage', 'dfgt-nograph', 'dfgt', 'dfgt+pf', 'dfgt+pf+il', 'dfgt+port+fine', 'dfgt+port+fine+il', 'dfgt+ile+pf+il',
          'M1+hs', 'M1+il+hs', 'dfgt+port+fine+il+hs',
          'dfgt+ile',          # S3：兼任作为边（dfgt+ile − M_e）
          'M1+il-lenient',     # 敏感性：兼任宽松版（与 M1 比较）
          'M1-noindmkt']       # 敏感性：M1 不含行业和市场
SRC = ['paper_confirm.py', 'paper_dev.py', 'paper_models.py', 'dev_models.py', 'paper_data.py', 'pair_feats.py', 'interlock_feats.py', 'hyslr_feats.py', 'build_v3.py', 'check_v3.py', 'check_v3_regress.py']

ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['run']); ap.add_argument('--dry-run', action='store_true')
ap.add_argument('--data', default='data_v3x', help='数据目录（相对 linkpred/）'); ap.add_argument('--feat', default='out_dev', help='点对特征文件目录（相对 linkpred/）')
ap.add_argument('--out', required=True, help='输出目录（相对 linkpred/），必须不存在或为空'); ap.add_argument('--device', default='cuda')
ap.add_argument('--models', default='', help='只跑这些模型（逗号分隔，演练时用）；正式运行必须跑全部')
A = ap.parse_args()
Y = DRY if A.dry_run else CONFIRM; L = Path(__file__).resolve().parents[1]; OUT = L / A.out
if Y['test'] == 2024 or Y['val'] >= Y['test'] or max(Y['train']) >= Y['val']: sys.exit('年份设置不合法：测试年不能是 2024，且须 训练 < 验证 < 测试')
if not A.dry_run and (Y['test'] != 2025 or A.models): sys.exit('正式运行：测试年必须是 2025，且必须跑全部模型')
if OUT.exists() and (any(OUT.glob('scores_*.csv')) or any(OUT.glob('val_scores_*.csv')) or (OUT / 'manifest.json').exists()):
    sys.exit(f'{OUT} 中已有分数文件或 manifest.json，拒绝运行（确认测试只运行一次）')

# paper_dev.py 在导入时解析命令行；给它一个空命令，并把输出目录、种子数、设备传进去
sys.path.insert(0, str(Path(__file__).resolve().parent))
argv0 = sys.argv; sys.argv = ['paper_dev.py', 'none', '--out', A.out, '--seeds', str(len(SEEDS)), '--device', A.device]
import paper_dev as PD
sys.argv = argv0
PD.D = L / A.data; PD.FE = L / A.feat; PD.GRAPH_YEARS = range(2019, Y['test'] + 1); PD.PF_ZERO_FROM = Y['test'] + 1
torch = PD.torch

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git_head():
    try: return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=L, capture_output=True, text=True).stdout.strip()
    except Exception: return ''
def git_dirty():
    try: return subprocess.run(['git', 'status', '--porcelain', '--', 'src'], cwd=L, capture_output=True, text=True).stdout.strip()
    except Exception: return ''

X, N, G = PD.load()
years = list(Y['train']) + [Y['val'], Y['test']]
for t in years:
    if not (X.fy == t).any():
        if OUT.exists() and not any(OUT.iterdir()): OUT.rmdir()   # paper_dev 导入时会建输出目录；检查不通过时删掉空目录
        sys.exit(f'数据中没有 fy = {t} 的点对')
models = A.models.split(',') if A.models else MODELS
OUT.mkdir(parents=True, exist_ok=True)   # 年份和数据检查都通过后才建输出目录
def env():
    import platform, sklearn, scipy
    v = dict(python=sys.version.split()[0], executable=sys.executable, prefix=sys.prefix, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__)
    if torch is not None:
        v.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
        try: import torch_geometric; v['torch_geometric'] = torch_geometric.__version__
        except ImportError: pass
    return v
man = dict(mode='dry-run' if A.dry_run else 'confirm', years=Y, seeds=SEEDS, models=models, command=' '.join(sys.argv), started=time.strftime('%Y-%m-%d %H:%M:%S %z'), environment=env(),
           git_head=git_head(), git_dirty_src=git_dirty(), source_sha256={f: sha(Path(__file__).resolve().parent / f) for f in SRC},
           data_sha256={p.name: sha(p) for p in sorted(PD.D.glob('*.csv'))},
           feature_sha256={p.name: sha(p) for p in sorted(PD.FE.glob('*_feats*.csv'))}, files={})
def save_manifest(): json.dump(man, open(OUT / 'manifest.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
save_manifest()

# 第 2.4 节的覆盖率门槛：全部点对中两端都有可用数据的比例。门槛 = 三个开发验证年（2021–2023）最低值的一半（兼任 .643 → .32；最大股东 .638 → .32）
COV = dict(interlock=('interlock_feats.csv', 'il_has_a', 'il_has_o', .643 / 2), hyslr=('hyslr_feats.csv', 'hs_has_a', 'hs_has_o', .638 / 2))
for k, (f, a_, b_, thr) in COV.items():
    F = pd.read_csv(PD.FE / f, index_col='row', usecols=['row', a_, b_]).reindex(X.index); c = ((F[a_] == 1) & (F[b_] == 1)).groupby(X.fy).mean()
    man.setdefault('coverage', {})[k] = dict(threshold=round(thr, 4), by_fy={int(t): round(float(v), 4) for t, v in c.items()},
                                           test_year_ok=bool(c.get(Y['test'], 0) >= thr))
    print('coverage', k, man['coverage'][k]['by_fy'], 'test year ok' if man['coverage'][k]['test_year_ok'] else f'TEST YEAR BELOW THRESHOLD {thr:.3f}: dependent tests are "not interpretable" (A8 sec. 2.4)', flush=True)
save_manifest()

VA, TE = X[X.fy == Y['val']], X[X.fy == Y['test']]; cols = ['pair_id', 'fy', 'anchor', 'other', 'label']
for v in models:
    model, opt = PD.VARIANTS[v]; t0 = time.time(); Gv = PD.add_interlock_edges(G, N, opt['ile']) if opt.get('ile') else G
    Ys = PD.node_tables(N, Gv, years, Y['train'], opt.get('target', False))
    if model == 'lr':
        sv, st = PD.run_lr(X, Ys, Y['train'], Y['val'], opt, te_year=Y['test']); per_v = per_t = {}
    else:
        res = {sd: PD.run_gnn(X, Ys, Y['train'], Y['val'], model, opt, sd, te_year=Y['test']) for sd in SEEDS}
        per_v = {f'score_s{sd}': r[0] for sd, r in res.items()}; per_t = {f'score_s{sd}': r[1] for sd, r in res.items()}
        sv, st = np.mean(list(per_v.values()), 0), np.mean(list(per_t.values()), 0)
    fv = OUT / ('val_' + PD.sfile(v).name); VA[cols].assign(fold='V', score=sv, **per_v).to_csv(fv, index=False)
    man['files'][fv.name] = sha(fv); save_manifest()   # 验证分数先落盘并记录哈希（第 9 节：冻结后再给测试年打分）
    ft = OUT / PD.sfile(v).name; TE[cols].assign(fold='T', score=st, **per_t).to_csv(ft, index=False)
    man['files'][ft.name] = sha(ft); man.setdefault('val_auc', {})[v] = round(PD.fast_auc(VA, sv), 4); man.setdefault('seconds', {})[v] = round(time.time() - t0, 1); save_manifest()
    print(v, 'val', man['val_auc'][v], f"{man['seconds'][v]}s", flush=True)   # 只打印验证 AUC；测试结果只由 paper_dev.py eval 计算
man['finished'] = time.strftime('%Y-%m-%d %H:%M:%S %z'); save_manifest(); print('done; manifest:', OUT / 'manifest.json')
