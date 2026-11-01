"""Fetch otrCprInvstmntSttus (11011) for every universe firm-year (FY order given), resumable via raw cache.
Usage: python invest_panel.py --years 2023 2022 2021 2020 2019 2024 2025 -> data/interim/invest_panel_raw.csv when all done.
Stops cleanly on daily budget (DailyLimitReached); rerun to resume."""
import argparse, csv, os
from concurrent.futures import ThreadPoolExecutor
from dart_client import DartClient, DailyLimitReached
from pipeline import INTERIM, PROC
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))

def main(years):
    c = DartClient(); U = [r for r in rd(PROC / "universe_firms.csv") if r["status_now"] != "konex_now"]
    out = []
    for y in years:
        codes = sorted(r["corp_code"] for r in U if str(y) in r["fy_with_annual_report"].split())
        f = lambda k: c.get_json("otrCprInvstmntSttus.json", {"corp_code": k, "bsns_year": str(y), "reprt_code": "11011"})
        with ThreadPoolExecutor(int(os.environ.get("DART_WORKERS", "10"))) as ex:
            js = list(ex.map(f, codes))
        for k, j in zip(codes, js):
            l = j.get("list") or []
            out += [dict(it, src_corp_code=k, bsns_year=y, api_status=j.get("status")) for it in l] or [{"src_corp_code": k, "bsns_year": y, "api_status": j.get("status")}]
        print("FY", y, "firms", len(codes), "calls today", c.calls_today(), flush=True)
    keys = sorted({k for r in out for k in r})
    with open(INTERIM / "invest_panel_raw.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=keys); w.writeheader(); w.writerows(out)
    print("written", len(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--years", type=int, nargs="+", required=True)
    try: main(ap.parse_args().years)
    except DailyLimitReached as e: print("STOPPED (resume tomorrow; cache keeps progress):", e)
