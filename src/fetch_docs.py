"""Fetch document.xml for the ORIGINAL FY{fy} 사업보고서 (first non-correction A001 filing on/before C) of firms,
priority: baseline test isolated (1,069) -> connected without financial neighbour (53) + their neighbours -> rest of universe.
Stored in data/raw/versions/<rcept_no>/document.zip (never overwritten). Stops at DART_STOP_AT calls (default 17,900)."""
import os, shutil, sys, pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dart_client import DartClient, DailyLimitReached, cache_key
R = Path(__file__).resolve().parents[1]; B = Path('/workspace/baseline-from-jack/enterprise-baseline')
FY, C = '2024', '20250430'; STOP = int(os.environ.get('DART_STOP_AT', '17900'))

def originals():
    A = pd.read_csv(R / 'data/interim/filings_A001.csv', dtype=str).drop_duplicates('rcept_no')
    a = A[A.report_nm.str.match(r'^사업보고서\s*\(' + FY + r'\.12\)') & A.rcept_dt.le(C)]
    return a.sort_values('rcept_no').drop_duplicates('corp_code').set_index('corp_code').rcept_no

def main():
    P = pd.read_csv(B / 'model-panel.csv', dtype=str); T = P[P.split.eq('test')]
    E = pd.read_csv(B / 'graphs/edges-2024-asof-20250430.csv', dtype=str)
    iso = list(T[T.degree.eq('0')].corp_code); nf = list(T[T.degree.ne('0') & T.neighbor_financial_count.eq('0')].corp_code)
    nb = list(E[E.src_corp_code.isin(nf)].target) + list(E[E.target.isin(nf)].src_corp_code)
    U = pd.read_csv(R / 'data/processed/universe_firms.csv', dtype=str)
    order = list(dict.fromkeys(iso + nf + nb + list(U.corp_code)))
    o = originals(); rs = [o[k] for k in order if k in o.index]
    print('firms', len(order), 'with original by C', len(rs), flush=True)
    c = DartClient()
    def one(r):
        dst = R / 'data/raw/versions' / r / 'document.zip'
        if dst.exists(): return
        if c.calls_today() >= STOP: raise DailyLimitReached('stop-at reserve reached')
        p = {'rcept_no': r}; c.get('document.xml', p, ext='zip'); dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix('.part'); shutil.copyfile(c.raw_dir / 'document' / f"{cache_key('document.xml', p)}.zip", tmp); tmp.rename(dst)
    with ThreadPoolExecutor(int(os.environ.get('DART_WORKERS', '8'))) as ex:
        for i, _ in enumerate(ex.map(one, rs), 1):
            if i % 200 == 0: print(i, 'calls today', c.calls_today(), flush=True)
    print('done', len(rs))

if __name__ == '__main__':
    try: main()
    except DailyLimitReached as e: print('STOPPED (resume after 00:00 KST):', e)
