"""Fetch an annual-report (11011) section for every universe firm-year, resumable via raw cache.
Default endpoint exctvSttus (임원현황, item 3: director/officer interlocks); --endpoint hyslrSttus (최대주주현황, item 4).
--endpoint majorstock (대량보유 상황보고, 5% holders): not per year; one call per firm that has an annual report in any of --years,
returning all its 5% reports (as-of filtering by rcept_dt is done later, in linkpred/src/majorstock_feats.py).
Only dev years are needed now: FY 2019-2023.
Usage: python exec_panel.py --years 2023 2022 2021 2020 2019 [--endpoint hyslrSttus]
       -> data/interim/exec_panel_raw.csv (exctvSttus) or data/interim/hyslr_panel_raw.csv (hyslrSttus) when all done.
Stops cleanly on daily budget (DailyLimitReached); rerun to resume."""
import argparse, csv, os
from concurrent.futures import ThreadPoolExecutor
from dart_client import DartClient, DailyLimitReached
from pipeline import INTERIM, PROC
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
OUT = {"exctvSttus": "exec_panel_raw.csv", "hyslrSttus": "hyslr_panel_raw.csv", "majorstock": "majorstock_raw.csv"}

def main(years, endpoint):
    c = DartClient(); U = [r for r in rd(PROC / "universe_firms.csv") if r["status_now"] != "konex_now"]
    out = []
    if endpoint == "majorstock":
        codes = sorted(r["corp_code"] for r in U if set(map(str, years)) & set(r["fy_with_annual_report"].split()))
        with ThreadPoolExecutor(int(os.environ.get("DART_WORKERS", "10"))) as ex:
            js = list(ex.map(lambda k: c.get_json("majorstock.json", {"corp_code": k}), codes))
        for k, j in zip(codes, js):
            out += [dict(it, src_corp_code=k, api_status=j.get("status")) for it in (j.get("list") or [])] or [{"src_corp_code": k, "api_status": j.get("status")}]
        print("majorstock firms", len(codes), "calls today", c.calls_today(), flush=True)
        years = []
    for y in years:
        codes = sorted(r["corp_code"] for r in U if str(y) in r["fy_with_annual_report"].split())
        f = lambda k: c.get_json(f"{endpoint}.json", {"corp_code": k, "bsns_year": str(y), "reprt_code": "11011"})
        with ThreadPoolExecutor(int(os.environ.get("DART_WORKERS", "10"))) as ex:
            js = list(ex.map(f, codes))
        for k, j in zip(codes, js):
            l = j.get("list") or []
            out += [dict(it, src_corp_code=k, bsns_year=y, api_status=j.get("status")) for it in l] or [{"src_corp_code": k, "bsns_year": y, "api_status": j.get("status")}]
        print("FY", y, "firms", len(codes), "calls today", c.calls_today(), flush=True)
    keys = sorted({k for r in out for k in r})
    with open(INTERIM / OUT[endpoint], "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=keys); w.writeheader(); w.writerows(out)
    print("written", len(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--endpoint", default="exctvSttus", choices=sorted(OUT)); a = ap.parse_args()
    try: main(a.years, a.endpoint)
    except DailyLimitReached as e: print("STOPPED (resume tomorrow; cache keeps progress):", e)
