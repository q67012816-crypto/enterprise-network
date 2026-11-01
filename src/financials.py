"""Fetch fnlttMultiAcnt (주요계정, up to 100 corps per call) for universe firms, FY range, annual report (11011).
Usage: python financials.py --from 2018 --to 2025 -> data/interim/fin_multi_raw.csv (one row per corp/year/fs_div/account)
Note: the API returns the LATEST filed version (corrections overwrite); as-of-C versions are not available (limitation)."""
import argparse, csv
from dart_client import DartClient, DailyLimitReached
from pipeline import INTERIM, PROC
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))

def main(y0, y1):
    c = DartClient(); firms = [r for r in rd(PROC / "universe_firms.csv") if r["status_now"] != "konex_now"]
    out, st = [], {}
    for y in range(y0, y1 + 1):
        codes = sorted(r["corp_code"] for r in firms if str(y) in r["fy_with_annual_report"].split())
        for i in range(0, len(codes), 100):
            j = c.get_json("fnlttMultiAcnt.json", {"corp_code": ",".join(codes[i:i + 100]), "bsns_year": str(y), "reprt_code": "11011"})
            st[(y, j.get("status"))] = st.get((y, j.get("status")), 0) + 1
            for it in j.get("list") or []:
                out.append({k: it.get(k, "") for k in ("corp_code", "bsns_year", "rcept_no", "fs_div", "sj_div", "account_nm",
                                                        "thstrm_dt", "thstrm_amount", "frmtrm_amount", "currency")})
        print(y, len(codes), "rows", len(out), "calls", c.calls_today(), flush=True)
    with open(INTERIM / "fin_multi_raw.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("status per (year,status):", st)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--from", dest="y0", type=int, default=2018); ap.add_argument("--to", dest="y1", type=int, default=2025)
    a = ap.parse_args()
    try: main(a.y0, a.y1)
    except DailyLimitReached as e: print("STOPPED (resume tomorrow):", e)
