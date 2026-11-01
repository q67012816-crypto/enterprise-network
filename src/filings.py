"""Fetch disclosure index (list.json) for given pblntf_detail_ty over a date range, 3-month windows (API limit
without corp_code), last_reprt_at=N so originals AND corrections ([기재정정] etc.) are kept with their rcept_dt.
Usage: python filings.py --ty A001 --start 2019-01-01 --end 2026-10-31  -> data/interim/filings_<ty>.csv (resumable via cache)"""
import argparse, csv, datetime as dt
from dart_client import DartClient, DailyLimitReached
from pipeline import INTERIM

def windows(start, end):
    s = start
    while s <= end:
        m = s.month + 3; y = s.year + (m - 1) // 12; m = (m - 1) % 12 + 1
        e = min(dt.date(y, m, 1) - dt.timedelta(days=1), end); yield s, e; s = e + dt.timedelta(days=1)

def main(ty, start, end):
    c = DartClient(); rows = []
    for s, e in windows(start, end):
        page = 1
        while True:
            j = c.get_json("list.json", {"bgn_de": s.strftime("%Y%m%d"), "end_de": e.strftime("%Y%m%d"), "pblntf_detail_ty": ty,
                                         "last_reprt_at": "N", "page_count": "100", "page_no": str(page)})
            rows += [dict(it, window=f"{s}~{e}") for it in (j.get("list") or [])]
            if j.get("status") != "000" or page >= int(j.get("total_page") or 1): break
            page += 1
        print(ty, s, e, "rows so far", len(rows), "calls today", c.calls_today(), flush=True)
    seen, out = set(), []
    for r in rows:
        if r["rcept_no"] not in seen: seen.add(r["rcept_no"]); out.append(r)
    with open(INTERIM / f"filings_{ty}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in out for k in r})); w.writeheader(); w.writerows(out)
    print("written", len(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ty", required=True); ap.add_argument("--start", default="2019-01-01"); ap.add_argument("--end", default="2026-10-31")
    a = ap.parse_args()
    try: main(a.ty, dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end))
    except DailyLimitReached as e: print("STOPPED (resume tomorrow, cache keeps progress):", e)
