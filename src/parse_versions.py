"""Parse every data/raw/versions/<rcept_no>/document.zip (+ fnlttXbrl.zip hash) into:
data/processed/doc_financials.csv, doc_invest_rows.csv, name_history.csv. Offline, deterministic."""
import hashlib, sys, zipfile, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from parse_document import parse_xbrl, main_xml, parse, company_name, invest_section, rename_table
R = Path(__file__).resolve().parents[1]
A = pd.read_csv(R / 'data/interim/filings_A001.csv', dtype=str).drop_duplicates('rcept_no').set_index('rcept_no')

def one(d):
    r = d.name; meta = A.loc[r] if r in A.index else None
    fy = meta.report_nm[meta.report_nm.find('(') + 1:][:4] if meta is not None else ''
    rec = dict(rcept_no=r, corp_code=meta.corp_code if meta is not None else '', fy=fy,
               rcept_dt=meta.rcept_dt if meta is not None else '', report_nm=meta.report_nm if meta is not None else '')
    xb = d / 'fnlttXbrl.zip'
    if xb.exists():
        try:
            z = zipfile.ZipFile(xb); n = [x for x in z.namelist() if x.endswith('.xbrl')]
            xx = z.read(n[0]) if n else b''
            rec['xbrl_sha1'] = hashlib.sha1(xx).hexdigest() if n else 'no_xbrl_member'
            if n and fy.isdigit(): rec.update({'xbrl_' + k: v for k, v in parse_xbrl(xx.decode('utf-8', 'replace'), int(fy)).items()})
        except zipfile.BadZipFile:
            rec['xbrl_sha1'] = 'not_zip:' + xb.read_bytes()[:120].decode('utf-8', 'replace').replace('\n', ' ')
    inv, names = [], []
    doc = d / 'document.zip'
    if not doc.exists(): rec['doc_status'] = 'missing'; return rec, inv, names
    try: t = main_xml(doc)
    except zipfile.BadZipFile: rec['doc_status'] = 'not_zip'; return rec, inv, names
    if t is None: rec['doc_status'] = 'no_main_xml'; return rec, inv, names
    rec['doc_status'] = 'ok'
    code, nm = company_name(t); rec['doc_corp_code'] = code; rec['doc_company_name'] = nm
    if fy.isdigit():
        v, flags = parse(t, int(fy)); rec.update(v); rec['parse_flags'] = ';'.join(flags)
    rows, found, none = invest_section(t); rec['invest_section'] = found; rec['invest_says_none'] = none; rec['invest_rows'] = len(rows)
    inv = [dict(rcept_no=r, corp_code=rec['corp_code'], fy=fy, rcept_dt=rec['rcept_dt'], **x) for x in rows]
    if nm: names.append(dict(corp_code=rec['corp_code'], name=nm, name_effective_from=rec['rcept_dt'], change_date='',
                             source='COMPANY-NAME as filed', rcept_no=r, source_verified='Y'))
    for ds, b, a in rename_table(t):
        names.append(dict(corp_code=rec['corp_code'], name=b, name_effective_from='00000000', change_date=ds,
                          source='상호의 변경 table: before', rcept_no=r, source_verified='Y'))
        names.append(dict(corp_code=rec['corp_code'], name=a, name_effective_from=ds or '00000000', change_date=ds,
                          source='상호의 변경 table: after', rcept_no=r, source_verified='Y'))
    return rec, inv, names

if __name__ == '__main__':
    dirs = sorted(p for p in (R / 'data/raw/versions').iterdir() if p.is_dir())
    with ProcessPoolExecutor(8) as ex: res = list(ex.map(one, dirs, chunksize=20))
    pd.DataFrame([r for r, _, _ in res]).to_csv(R / 'data/processed/doc_financials.csv', index=False)
    pd.DataFrame([x for _, i, _ in res for x in i]).to_csv(R / 'data/processed/doc_invest_rows.csv', index=False)
    pd.DataFrame([x for _, _, n in res for x in n]).drop_duplicates().to_csv(R / 'data/processed/name_history.csv', index=False)
    print('docs', len(res))
