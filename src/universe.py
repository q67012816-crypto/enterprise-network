"""Historical listing universe (Phase 2, Step 1f). Offline: uses corp_codes.csv, company_overview.csv,
filings_A001.csv, filings_I003.csv. Usage: python universe.py
Outputs:
  data/processed/universe_firms.csv  one row per ever-stock-coded firm with an A001 filing 2019-2026:
      current corp_cls, fiscal years with an original 사업보고서, delisting evidence (I003), classification
  data/processed/delisting_review.csv  firms not currently listed that have delisting-keyword notices, classified
  reports/universe.json  counts
Rules (all approximations; exact listing/delisting dates are NOT available from OpenDART = 未验证):
  - candidate = has stock_code in corpCode AND filed an A001 사업보고서 for some FY 2018-2025.
  - currently listed = company.json corp_cls in {Y,K}.
  - formerly listed (likely) = corp_cls E now, stock_code present. Konex (N) excluded from KOSPI/KOSDAQ universe.
  - delisting classification (I003 report_nm):
      cause_delisting_completed: 정리매매 개시/재개 or 상장폐지 결정 notice exists
      voluntary_or_merger_keyword: 자진 상장폐지 / 합병 / 지주회사 in notices
      cause_notice_only: only 상장폐지 사유 발생/우려 notices -> outcome 未验证
      no_delisting_notice: formerly listed with no keyword notice (merger/transfer/other) -> reason 未验证
  - approx last trading date = date of last 정리매매 notice (+~7 trading days; 未验证)."""
import csv, json, re
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; P = ROOT / "data/processed"; I = ROOT / "data/interim"
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
ov = {r["corp_code"]: r for r in rd(I / "company_overview.csv")}
A = rd(I / "filings_A001.csv"); E3 = rd(I / "filings_I003.csv")
fy = defaultdict(set); orig_dates = defaultdict(dict)
for r in A:
    m = re.search(r"사업보고서\s*\((\d{4})\.(\d{2})\)", r["report_nm"])
    if not m or "연장신고" in r["report_nm"]: continue
    y = int(m.group(1)) if int(m.group(2)) >= 1 else None
    # fiscal year label: FY = year of period end (matches bsns_year convention, 未验证 for non-Dec)
    fy[r["corp_code"]].add(y)
    if not r["report_nm"].startswith("["):
        orig_dates[r["corp_code"]].setdefault(y, r["rcept_dt"])
notes = defaultdict(list)
for r in E3:
    if re.search(r"상장폐지|정리매매", r["report_nm"]): notes[r["corp_code"]].append(r)
def classify(c):
    ns = notes.get(c, []); t = " ".join(n["report_nm"] for n in ns)
    if re.search(r"정리매매\s*(개시|재개|절차\s*재개)|상장폐지\s*결정", t): k = "cause_delisting_completed"
    elif re.search(r"자진\s*상장폐지|합병|지주회사", t): k = "voluntary_or_merger_keyword"
    elif ns: k = "cause_notice_only"
    else: k = "no_delisting_notice"
    last = max((n["rcept_dt"] for n in ns if "정리매매" in n["report_nm"]), default="")
    return k, last, len(ns)
rows = []
for c, o in ov.items():
    if not fy.get(c): continue
    years = sorted(y for y in fy[c] if y and 2018 <= y <= 2025)
    if not years: continue
    k, last, n = classify(c)
    status = "listed_now" if o["corp_cls"] in ("Y", "K") else ("konex_now" if o["corp_cls"] == "N" else "not_listed_now")
    rows.append({"corp_code": c, "corp_name": o["corp_name"], "stock_code": o["stock_code"], "corp_cls_now": o["corp_cls"],
                 "status_now": status, "induty_code": o["induty_code"], "acc_mt": o["acc_mt"], "fy_with_annual_report": " ".join(map(str, years)),
                 "first_fy": years[0], "last_fy": years[-1],
                 "delisting_class": k if status == "not_listed_now" else "", "last_jeongni_notice": last, "n_delisting_notices": n,
                 "orig_filing_dates": json.dumps({str(y): d for y, d in sorted(orig_dates[c].items()) if y}, ensure_ascii=False)})
with open(P / "universe_firms.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
rev = [r for r in rows if r["status_now"] != "listed_now" and r["n_delisting_notices"]]
with open(P / "delisting_review.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rev)
cnt = {"firms_in_universe": len(rows), "by_status_now": dict(Counter(r["status_now"] for r in rows)),
       "not_listed_now_by_class": dict(Counter(r["delisting_class"] for r in rows if r["status_now"] == "not_listed_now")),
       "not_listed_now_last_fy": dict(sorted(Counter(r["last_fy"] for r in rows if r["status_now"] == "not_listed_now").items())),
       "firm_years_by_fy_listed_or_former": {y: sum(str(y) in r["fy_with_annual_report"].split() for r in rows if r["status_now"] != "konex_now") for y in range(2018, 2026)},
       "delisting_review_rows": len(rev), "delisting_review_by_class": dict(Counter(r["delisting_class"] or r["status_now"] for r in rev))}
(ROOT / "reports/universe.json").write_text(json.dumps(cnt, ensure_ascii=False, indent=1)); print(json.dumps(cnt, ensure_ascii=False, indent=1))
