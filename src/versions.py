"""STEP1(a): version index + per-rcept_no downloads for late-label samples.
Targets: 314 excluded validation (corp, FY2024) from baseline excluded-validation-labels.csv, plus
late-label training candidates (feature FY t in 2021/2022 whose FY t+1 OFS latest disclosure > 20240430; superset of baseline's 77).
For each target firm-year: all A001 filings whose report_nm period is (FY.12) -> data/processed/versions_index.csv.
Downloads document.xml and fnlttXbrl.xml(11011) per rcept_no via cached client, then copies into
data/raw/versions/<rcept_no>/ (never overwrites an existing file). Budget-safe; rerun to resume."""
import shutil, sys, pandas as pd
from pathlib import Path
from dart_client import DartClient, DailyLimitReached, cache_key
R = Path(__file__).resolve().parents[1]; BASE = Path('/workspace/baseline-from-jack/enterprise-baseline')

def targets():
    X = pd.read_csv(BASE / 'excluded-validation-labels.csv', dtype=str)
    t = [dict(corp_code=k, fy='2024', group='validation_314') for k in X.corp_code]
    A = pd.read_csv(R / 'data/interim/filings_A001.csv', dtype=str).drop_duplicates('rcept_no')
    F = pd.read_csv(R / 'data/interim/fin_multi_raw.csv', dtype=str, usecols=['corp_code', 'bsns_year', 'fs_div', 'rcept_no'])
    F = F[F.fs_div.eq('OFS')].copy(); F['d'] = F.rcept_no.map(A.set_index('rcept_no').rcept_dt)
    d = F.groupby(['corp_code', 'bsns_year']).d.max().unstack()
    for y, C in [(2021, '20220430'), (2022, '20230430')]:
        ok = d[d[str(y)] <= C]
        t += [dict(corp_code=k, fy=str(y + 1), group='train_late_candidate') for k in ok[ok[str(y + 1)] > '20240430'].index]
    return pd.DataFrame(t), A

def main():
    T, A = targets()
    A['fy'] = A.report_nm.str.extract(r'\((\d{4})\.12\)')[0]
    V = T.merge(A[['corp_code', 'fy', 'rcept_no', 'rcept_dt', 'report_nm']], on=['corp_code', 'fy'], how='left')
    V['version_rank'] = V.groupby(['corp_code', 'fy']).rcept_no.rank(method='first')
    V.to_csv(R / 'data/processed/versions_index.csv', index=False)
    print('targets', len(T), 'versions', V.rcept_no.notna().sum(), 'no filing', V.rcept_no.isna().sum(), flush=True)
    c = DartClient(); n = 0
    import os
    from concurrent.futures import ThreadPoolExecutor
    def one(r):
        out = R / 'data/raw/versions' / r; out.mkdir(parents=True, exist_ok=True)
        for ep, p, name in [('document.xml', {'rcept_no': r}, 'document.zip'),
                            ('fnlttXbrl.xml', {'rcept_no': r, 'reprt_code': '11011'}, 'fnlttXbrl.zip')]:
            dst = out / name
            if dst.exists(): continue
            c.get(ep, p, ext='zip')
            src = c.raw_dir / ep.rsplit('.', 1)[0] / f"{cache_key(ep, p)}.zip"
            tmp = dst.with_suffix('.part'); shutil.copyfile(src, tmp); tmp.rename(dst)
    rs = list(V.rcept_no.dropna().unique())
    with ThreadPoolExecutor(int(os.environ.get('DART_WORKERS', '8'))) as ex:
        for i, _ in enumerate(ex.map(one, rs), 1):
            if i % 100 == 0: print(i, 'versions; calls today', c.calls_today(), flush=True)
    print('done', len(rs), 'calls today', c.calls_today())

if __name__ == '__main__':
    try: main()
    except DailyLimitReached as e: print('STOPPED, resume after 00:00 KST:', e)
