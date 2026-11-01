"""Label feasibility measurement (NO modeling). Usage: python labels.py --year 2025
Outputs data/processed/labels_*.csv and reports/labels_<year>.json."""
import os, argparse, json, re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dart_client import DartClient
from pipeline import INTERIM, PROC, _read, _write
from build import pct
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "reports"

QUARTERS = lambda y: [(f"{y}0101", f"{y}0331"), (f"{y}0401", f"{y}0630"), (f"{y}0701", f"{y}0930"), (f"{y}1001", f"{y}1231")]
B_KW = {"default_dfOcr": "부도발생", "bsn_suspension_bsnSp": "영업정지", "rehab_ctrcvsBgrq": "회생절차개시신청", "dissolution_dsRsOcr": "해산사유발생"}
I_KW = {"mgmt_designation": r"관리종목\s*지정", "delisting_cause": r"상장폐지\s*사유\s*발생|상장폐지\s*결정|상장폐지$|정리매매",
        "trading_halt": r"매매거래정지(?!기간변경)", "listing_eligibility_review": r"상장적격성"}

def list_all(c, ty, bgn, end):
    out, page = [], 1
    while True:
        j = c.get_json("list.json", {"bgn_de": bgn, "end_de": end, "pblntf_detail_ty": ty, "page_count": "100", "page_no": str(page)})
        out += j.get("list") or []
        if j.get("status") != "000" or page >= int(j.get("total_page") or 1): return out
        page += 1

def main(year):
    c = DartClient(); listed = _read(INTERIM / "listed.csv"); codes = {r["corp_code"] for r in listed}
    res = {"year": year, "n_listed": len(listed)}
    # 1) audit opinion (one call returns 당기/전기/전전기)
    with ThreadPoolExecutor(int(os.environ.get("DART_WORKERS","10"))) as ex:
        js = list(ex.map(lambda r: c.get_json("accnutAdtorNmNdAdtOpinion.json", {"corp_code": r["corp_code"], "bsns_year": str(year), "reprt_code": "11011"}), listed))
    rows, st = [], Counter()
    for r, j in zip(listed, js):
        st[j.get("status")] += 1
        for it in j.get("list") or []:
            rows.append({"corp_code": r["corp_code"], "market": r["market"], "period": it.get("bsns_year"), "adt_opinion": (it.get("adt_opinion") or "").strip(), "adtor": it.get("adtor")})
    _write(PROC / "labels_audit_opinion.csv", rows)
    by = defaultdict(Counter)
    for x in rows: by[x["period"]][x["adt_opinion"]] += 1
    res["audit_status"] = dict(st); res["audit_opinion_values_by_period"] = {k: dict(v.most_common(25)) for k, v in by.items()}
    # 2) capital erosion from fnlttMultiAcnt (100 corps / call)
    cl = [r["corp_code"] for r in listed]; fin = []
    for i in range(0, len(cl), 100):
        j = c.get_json("fnlttMultiAcnt.json", {"corp_code": ",".join(cl[i:i+100]), "bsns_year": str(year), "reprt_code": "11011"})
        fin += j.get("list") or []
    code_by_stock = {r["stock_code"]: r["corp_code"] for r in listed}
    eq = defaultdict(dict)
    for f in fin:
        if f.get("account_nm") in ("자본총계", "자본금"):
            k = (code_by_stock.get(f.get("stock_code")), f.get("fs_div"))
            eq[k][f["account_nm"]] = pct(f.get("thstrm_amount"))
    ce = []
    for (code, fs), v in eq.items():
        if code and v.get("자본총계") is not None and v.get("자본금"):
            t, k = v["자본총계"], v["자본금"]
            ce.append({"corp_code": code, "fs_div": fs, "equity": t, "capital": k,
                       "partial_erosion": t < k, "erosion_ge50": t < 0.5 * k, "full_erosion": t <= 0})
    _write(PROC / "labels_capital_erosion.csv", ce)
    res["capital_erosion"] = {fs: {"n": sum(1 for x in ce if x["fs_div"] == fs), **{m: sum(1 for x in ce if x["fs_div"] == fs and x[m]) for m in ("partial_erosion", "erosion_ge50", "full_erosion")}} for fs in ("OFS", "CFS")}
    res["fin_corps_returned"] = len({f.get("stock_code") for f in fin})
    # 3) disclosure-search based events, calendar years year and year-1, report_nm keyword match
    ev = []
    for y in (year - 1, year):
        for ty, kw in (("B001", B_KW), ("I003", I_KW)):
            for b, e in QUARTERS(y):
                for it in list_all(c, ty, b, e):
                    nm = re.sub(r"\s+", "", it.get("report_nm", ""))
                    for lab, pat in kw.items():
                        if re.search(pat.replace(r"\s*", ""), nm):
                            ev.append({"label": lab, "cal_year": y, "corp_code": it["corp_code"], "listed_now": it["corp_code"] in codes,
                                       "corp_cls": it.get("corp_cls"), "report_nm": it.get("report_nm", "").strip(), "rcept_dt": it.get("rcept_dt")})
    _write(PROC / "labels_events.csv", ev)
    agg = defaultdict(lambda: defaultdict(set))
    for x in ev:
        if x["corp_cls"] in ("Y", "K"): agg[x["label"]][x["cal_year"]].add(x["corp_code"])
    res["events_unique_YK_corps"] = {l: {y: len(s) for y, s in v.items()} for l, v in agg.items()}
    res["calls_today_end"] = c.calls_today()
    (R / f"labels_{year}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(json.dumps(res, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--year", type=int, required=True); main(ap.parse_args().year)
