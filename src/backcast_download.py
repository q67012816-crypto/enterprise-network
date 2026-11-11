"""DEV_RESULTS 第 26.5 节：2015–2018 年回测的数据下载（只写到 data/interim/backcast/，不覆盖任何现有原始数据）。
API 密钥只从环境变量 OPENDART_API_KEY 读取（dart_client.py），不写入任何文件。可中断、可续跑（原始响应缓存）。
步骤：filings → company → universe → invest → exec → fin（按顺序，每步完成后写 CSV；已存在的 CSV 跳过）。
用法：python backcast_download.py [--steps filings company universe invest exec fin]"""
import argparse, csv, datetime as dt, json, os, re
from concurrent.futures import ThreadPoolExecutor
from dart_client import DartClient, DailyLimitReached
from pipeline import INTERIM
from filings import windows
OUT = INTERIM / 'backcast'; OUT.mkdir(exist_ok=True)
rd = lambda p: list(csv.DictReader(open(p, encoding='utf-8')))
def write(name, rows):
    keys = sorted({k for r in rows for k in r})
    with open(OUT / name, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print('written', name, len(rows), flush=True)

def step_filings(c):
    rows = []
    for s, e in windows(dt.date(2015, 1, 1), dt.date(2018, 12, 31)):
        page = 1
        while True:
            j = c.get_json('list.json', {'bgn_de': s.strftime('%Y%m%d'), 'end_de': e.strftime('%Y%m%d'), 'pblntf_detail_ty': 'A001', 'last_reprt_at': 'N', 'page_count': '100', 'page_no': str(page)})
            rows += [dict(it, window=f'{s}~{e}') for it in (j.get('list') or [])]
            if j.get('status') != '000' or page >= int(j.get('total_page') or 1): break
            page += 1
        print('A001', s, e, len(rows), 'calls', c.calls_today(), flush=True)
    seen, out = set(), []
    for r in rows:
        if r['rcept_no'] not in seen: seen.add(r['rcept_no']); out.append(r)
    write('filings_A001_2015_2018.csv', out)

def annual_fy(rows):
    """corp_code → 年报所属 FY 集合（与 build_v3 相同的口径：'사업보고서 (YYYY.MM)'，不含 연장신고）。"""
    d = {}
    for r in rows:
        nm = r.get('report_nm', '')
        if re.match(r'^사업보고서\s*\(', nm) and '연장신고' not in nm:
            m = re.search(r'사업보고서\s*\((\d{4})\.', nm)
            if m: d.setdefault(r['corp_code'], set()).add(int(m.group(1)))
    return d

def step_company(c):
    have = {r['corp_code'] for r in rd(INTERIM / 'company_overview.csv')}
    codes = sorted(set(annual_fy(rd(OUT / 'filings_A001_2015_2018.csv'))) - have)
    with ThreadPoolExecutor(int(os.environ.get('DART_WORKERS', '8'))) as ex:
        js = list(ex.map(lambda k: c.get_json('company.json', {'corp_code': k}), codes))
    write('company_overview_new.csv', [{k: j.get(k, '') for k in ('corp_code', 'corp_name', 'stock_name', 'stock_code', 'corp_cls', 'induty_code', 'jurir_no', 'bizr_no', 'acc_mt', 'status')} for j in js])

def step_universe(c):
    ov = {r['corp_code']: r for r in rd(INTERIM / 'company_overview.csv') + rd(OUT / 'company_overview_new.csv')}
    fy = annual_fy(rd(OUT / 'filings_A001_2015_2018.csv') + rd(INTERIM / 'filings_A001.csv'))
    rows = []
    for k, ys in fy.items():
        o = ov.get(k)
        if not o: continue
        cls = o.get('corp_cls', ''); ok = cls in ('Y', 'K') or (cls == 'E' and o.get('stock_code', '').strip())   # 与 universe.py 相同；KONEX（N）除外
        yy = sorted(y for y in ys if 2014 <= y <= 2018)
        if ok and yy: rows.append(dict(corp_code=k, corp_cls_now=cls, fy_with_annual_report=' '.join(map(str, yy))))
    write('universe_backcast.csv', rows)

def panel(c, endpoint, years, name):
    U = rd(OUT / 'universe_backcast.csv'); out = []
    for y in years:
        codes = sorted(r['corp_code'] for r in U if str(y) in r['fy_with_annual_report'].split())
        f = lambda k: c.get_json(f'{endpoint}.json', {'corp_code': k, 'bsns_year': str(y), 'reprt_code': '11011'})
        with ThreadPoolExecutor(int(os.environ.get('DART_WORKERS', '8'))) as ex: js = list(ex.map(f, codes))
        for k, j in zip(codes, js):
            l = j.get('list') or []
            out += [dict(it, src_corp_code=k, bsns_year=y, api_status=j.get('status')) for it in l] or [{'src_corp_code': k, 'bsns_year': y, 'api_status': j.get('status')}]
        print(endpoint, 'FY', y, 'firms', len(codes), 'calls', c.calls_today(), flush=True)
    write(name, out)

def step_fin(c):
    U = rd(OUT / 'universe_backcast.csv'); out = []
    for y in (2015, 2016, 2017):
        codes = sorted(r['corp_code'] for r in U if str(y) in r['fy_with_annual_report'].split())
        for i in range(0, len(codes), 100):
            j = c.get_json('fnlttMultiAcnt.json', {'corp_code': ','.join(codes[i:i + 100]), 'bsns_year': str(y), 'reprt_code': '11011'})
            for it in j.get('list') or []:
                out.append({k: it.get(k, '') for k in ('corp_code', 'bsns_year', 'rcept_no', 'fs_div', 'sj_div', 'account_nm', 'thstrm_dt', 'thstrm_amount', 'frmtrm_amount', 'currency')})
        print('fin', y, len(codes), 'rows', len(out), 'calls', c.calls_today(), flush=True)
    write('fin_multi_raw_2015_2017.csv', out)

STEPS = {'filings': (step_filings, 'filings_A001_2015_2018.csv'), 'company': (step_company, 'company_overview_new.csv'), 'universe': (step_universe, 'universe_backcast.csv'),
         'invest': (lambda c: panel(c, 'otrCprInvstmntSttus', (2018, 2017, 2016, 2015, 2014), 'invest_panel_raw_2014_2018.csv'), 'invest_panel_raw_2014_2018.csv'),
         'exec': (lambda c: panel(c, 'exctvSttus', (2018, 2017, 2016, 2015), 'exec_panel_raw_2015_2018.csv'), 'exec_panel_raw_2015_2018.csv'),
         'fin': (step_fin, 'fin_multi_raw_2015_2017.csv')}
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--steps', nargs='+', default=list(STEPS)); a = ap.parse_args(); c = DartClient()
    try:
        for s in a.steps:
            fn, f = STEPS[s]
            if (OUT / f).exists(): print('skip', s, '(exists)'); continue
            fn(c)
        print('ALL DONE', 'calls today', c.calls_today())
    except DailyLimitReached as e: print('STOPPED (daily budget; rerun tomorrow to resume from cache):', e)
