"""修订 A8.2（只报告，不判定）：确认运行中按正例首次取得日相对预测日 C_t 分组报告。规则与 DEV_RESULTS 第 21.1 节相同（date_rules.py）。
不修改任何 A8 冻结的文件；不重新训练，只读取冻结运行（out_confirm）的测试分数，并按其 manifest.json 核对 SHA-256。
必须在 paper_confirm.py 正式运行及其两条评估命令完成之后运行。

正式运行（2027 年）：
  python src/confirm_datesplit.py --data <2027 年的数据目录> --ref out_confirm
演练（只用开发年，与 paper_confirm.py --dry-run 相同的年份；不碰 2024→25）：
  python src/confirm_datesplit.py --dry-run --ref out_confirm_dryrun

步骤：
  1. 在本进程内执行 build_v3.py（运行时的版本，即 2027 年按 A8 第 5.3 节只改了年份常量的版本；参数 --first-year 2019），输出到仓库外的临时目录；
     重建的 pairs.csv 必须与 --data 中的 pairs.csv 逐字节相同，且与冻结运行 manifest.json 记录的 SHA-256 相同，否则停止。
  2. 测试年的每个正例按 date_rules.py 分类：after / before / unknown（细分 earlier、later、year_only、missing、no_row）。
  3. 用冻结运行的 scores_M1、scores_M1+il、scores_dfgt+port+fine、scores_dfgt+port+fine+il，计算每个正例的组内 AUC；
     度 = 测试年 graph_t.csv 上 Counter(src + dst)（与 paper_dev.evaluate 相同）。
  4. 在全样本和 after、before、unknown 子集上报告 M1+il − M1（低连接、孤立、全部；B = 10,000）和 C3′（B = 1,000）：
     点估计、双侧 95% 区间、单侧 95% 下限；numpy.default_rng(20261103)，在子集内按“折 × 投资方”聚类重抽（与第 21.1 节相同）。
输出：linkpred/out_confirm_datesplit（演练：out_confirm_datesplit_dryrun）/datesplit.json、positives.csv、manifest.json；目录已有结果时拒绝运行。"""
import argparse, collections, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

CONFIRM = dict(train=[2019, 2020, 2021, 2022, 2023], val=2024, test=2025)   # 与 paper_confirm.py 相同
DRY = dict(train=[2019, 2020, 2021], val=2022, test=2023)
COMPS = [('M1+il', 'M1', 10000), ('dfgt+port+fine+il', 'dfgt+port+fine', 1000)]
ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--data', default='data_v3x')
ap.add_argument('--ref', required=True); ap.add_argument('--scratch', default='')
A = ap.parse_args(); Y = DRY if A.dry_run else CONFIRM; T = Y['test']
if T == 2024: sys.exit('测试年不能是 2024')
L = Path(__file__).resolve().parents[1]; SRC = Path(__file__).resolve().parent; sys.path.insert(0, str(SRC))
DATA = L / A.data; REF = L / A.ref; OUT = L / ('out_confirm_datesplit' + ('_dryrun' if A.dry_run else ''))
SC = Path(A.scratch) if A.scratch else L.parent.parent / ('scratch_datesplit' + ('_dryrun' if A.dry_run else ''))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if (OUT / 'datesplit.json').exists(): sys.exit(f'{OUT} 已有结果，拒绝运行（只运行一次）')
rm = json.load(open(REF / 'manifest.json', encoding='utf-8'))
if 'finished' not in rm: sys.exit(f'{REF} 的冻结运行尚未完成')
if rm['years'] != Y: sys.exit(f'年份与冻结运行不一致：{rm["years"]}')
if rm['data_sha256'].get('pairs.csv') != sha(DATA / 'pairs.csv'): sys.exit('--data 的 pairs.csv 与冻结运行使用的数据不一致')
OUT.mkdir(exist_ok=True); SC.mkdir(parents=True, exist_ok=True)
def git_head():
    try: return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=L, capture_output=True, text=True).stdout.strip()
    except Exception: return ''
man = dict(mode='dry-run' if A.dry_run else 'confirm', years=Y, command=' '.join(sys.argv), started=time.strftime('%Y-%m-%d %H:%M:%S %z'), git_head=git_head(), ref=A.ref,
           source_sha256={f: sha(SRC / f) for f in ('confirm_datesplit.py', 'date_rules.py', 'build_v3.py')})

# 1. 重建出资行
argv0 = sys.argv; sys.argv = ['build_v3.py', '--out', str(SC), '--first-year', '2019']
ns = dict(__name__='__main__', __file__=str(SRC / 'build_v3.py'))
exec(compile((SRC / 'build_v3.py').read_text(encoding='utf-8'), str(SRC / 'build_v3.py'), 'exec'), ns)
sys.argv = argv0
man['rebuild_pairs_identical'] = sha(SC / 'pairs.csv') == sha(DATA / 'pairs.csv')
if not man['rebuild_pairs_identical']: json.dump(man, open(OUT / 'manifest.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False); sys.exit('重建的 pairs.csv 与 --data 不一致，停止')
from date_rules import row_date_cat, pair_rows, pair_date_class
ROWS, parse_date = ns['ROWS'], ns['parse_date']

# 2. 测试年正例分类
X = pd.read_csv(DATA / 'pairs.csv', dtype={'anchor': str, 'other': str, 'pair_id': str, 'directions': str})
POS = X[(X.label == 1) & (X.fy == T)].set_index('pair_id')
g = pd.read_csv(DATA / f'graph_{T}.csv', dtype=str); deg = collections.Counter(list(g.src) + list(g.dst))
rec = []
for pid, p in POS.iterrows():
    rr = pair_rows(ROWS, T, p.directions); dc = [row_date_cat(a, s, T, parse_date) for a, s in zip(rr.frst_acqs_de, rr.stlm_dt)]
    c = pair_date_class(dc); rec.append(dict(pair_id=pid, n_rows=len(rr), cls=c, main=c.split(':')[0], mindeg=min(deg[p.anchor], deg[p.other])))
R = pd.DataFrame(rec).set_index('pair_id')

# 3. 冻结运行的分数 → 每个正例的组内 AUC
def per_pos(f):
    S = pd.read_csv(f, dtype={'pair_id': str, 'anchor': str, 'other': str}); out = {}; fold = {}
    for pid, gg in S.groupby('pair_id'):
        s = gg.score.values[gg.label.values == 1][0]; n = gg.score.values[gg.label.values == 0]
        out[pid] = ((n < s).sum() + .5 * (n == s).sum()) / len(n); fold[pid] = gg.fold.iloc[0]
    return pd.Series(out), pd.Series(fold)
AUC = {}
for v in sorted({x for a, b, _ in COMPS for x in (a, b)}):
    f = REF / f'scores_{v}.csv'
    if sha(f) != rm['files'][f.name]: sys.exit(f'{f} 的 SHA-256 与冻结运行的 manifest.json 不一致')
    AUC[v], fold = per_pos(f); man.setdefault('ref_scores_sha256', {})[f.name] = rm['files'][f.name]
R['cluster'] = fold.reindex(R.index) + '-' + POS.anchor.reindex(R.index)
for v in AUC: R[v] = AUC[v].reindex(R.index)
R.to_csv(OUT / 'positives.csv')

# 4. 比较
GRP = {'all': lambda U: np.ones(len(U), bool), 'low': lambda U: (U.mindeg <= 1).values, 'iso': lambda U: (U.mindeg == 0).values}
def draws_for(U, B):
    cl = U.cluster.values; uc = np.unique(cl); ix = {c: np.where(cl == c)[0] for c in uc}; rng = np.random.default_rng(20261103)
    return [np.concatenate([ix[c] for c in rng.choice(uc, len(uc))]) for _ in range(B)]
def ci(d, m, dr):
    if m.sum() == 0: return None
    v = np.array([d[x][m[x]].mean() if m[x].any() else np.nan for x in dr])
    return [round(float(d[m].mean()), 4), round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4), round(float(np.nanpercentile(v, 5)), 4), int(m.sum())]
res = dict(counts={g_: R[GRP[g_](R)].cls.value_counts().to_dict() for g_ in GRP}, counts_main={g_: R[GRP[g_](R)].main.value_counts().to_dict() for g_ in GRP}, results={})
for sname in ('full', 'after', 'before', 'unknown'):
    sm = np.ones(len(R), bool) if sname == 'full' else (R.main == sname).values; U = R[sm]
    if len(U) == 0: res['results'][sname] = None; continue
    DR = {B: draws_for(U, B) for B in (1000, 10000)}
    for a, b, B in COMPS:
        d = (U[a] - U[b]).values; res['results'].setdefault(sname, {})[f'{a} − {b}'] = {g_: ci(d, GRP[g_](U), DR[B]) for g_ in GRP}
f_, a_ = res['results']['full']['M1+il − M1']['low'], (res['results'].get('after') or {}).get('M1+il − M1', {}).get('low')
res['retained_after_low'] = round(a_[0] / f_[0], 4) if a_ and f_ and f_[0] else None
json.dump(res, open(OUT / 'datesplit.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
man.update(finished=time.strftime('%Y-%m-%d %H:%M:%S %z'), output_sha256={f: sha(OUT / f) for f in ('datesplit.json', 'positives.csv')})
json.dump(man, open(OUT / 'manifest.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(res['counts'], ensure_ascii=False))
for s, v in res['results'].items():
    for k, x in (v or {}).items(): print(f'{s:8s} {k:36s}', '  '.join(f'{g_}: {y[0]:+.3f} [{y[1]:+.3f},{y[2]:+.3f}] lo{y[3]:+.3f} n{y[4]}' for g_, y in x.items() if y))
