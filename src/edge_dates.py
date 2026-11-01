"""Attach disclosure date to each directed edge via list.json A001 index (filings_A001.csv).
Usage: python edge_dates.py  -> data/processed/edges_dated.csv, reports/edge_dates.json"""
import csv, json, re
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; P = ROOT / "data/processed"; I = ROOT / "data/interim"
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
fil = {r["rcept_no"]: r for r in rd(I / "filings_A001.csv")}
edges = rd(P / "edges.csv"); out, st = [], {}
for e in edges:
    f = fil.get(e["rcept_no"]); y = e["bsns_year"]
    s = st.setdefault(y, Counter())
    dt_ = f["rcept_dt"] if f else ""
    s["edges"] += 1; s["found_in_index" if f else "not_in_index"] += 1
    if f:
        s["rcept_no_prefix_equals_rcept_dt" if e["rcept_no"][:8] == dt_ else "prefix_differs"] += 1
        s["filed_" + dt_[:4]] += 1
        corr = bool(re.match(r"\[(기재|첨부)정정|\[첨부추가", f["report_nm"]))
        s["is_correction_filing" if corr else "is_original_filing"] += 1
        s["on_or_before_Apr30_next_year" if dt_ <= f"{int(y)+1}0430" else "after_Apr30_next_year"] += 1
    out.append(dict(e, rcept_dt=dt_, report_nm=f["report_nm"] if f else "", date_source="list.json A001" if f else "rcept_no prefix (未验证)",
                    disclosure_date=dt_ or e["rcept_no"][:8]))
with open(P / "edges_dated.csv", "w", newline="", encoding="utf-8") as fo:
    w = csv.DictWriter(fo, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
allf = rd(I / "filings_A001.csv")
summ = {"edges_by_year": {y: dict(c) for y, c in st.items()},
        "A001_index": {"filings": len(allf), "rcept_no_prefix_equals_rcept_dt": sum(r["rcept_no"][:8] == r["rcept_dt"] for r in allf)}}
(ROOT / "reports/edge_dates.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1)); print(json.dumps(summ, ensure_ascii=False, indent=1))
