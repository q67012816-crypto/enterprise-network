"""Pipeline steps (Phase B). Each step is idempotent: cached raw responses are reused,
so re-running after an interruption or a DailyLimitReached resumes where it stopped.

Usage: python src/pipeline.py <step> [--year 2025] [--prev-year 2024]
Steps: corpcodes | listed | overview | invest | resolve | build | all
"""
from __future__ import annotations
import argparse, csv, io, json, re, sys, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from dart_client import DartClient, DailyLimitReached, ROOT

PROC = ROOT / "data/processed"
INTERIM = ROOT / "data/interim"  # derived but not for redistribution decisions; gitignored? see README
REPRT_ANNUAL = "11011"  # 사업보고서 (official guide)


def step_corpcodes(c: DartClient):
    """corpCode.xml -> data/interim/corp_codes.csv (all corps)."""
    z = zipfile.ZipFile(io.BytesIO(c.get("corpCode.xml", {}, ext="zip")))
    root = ET.fromstring(z.read(z.namelist()[0]))
    INTERIM.mkdir(parents=True, exist_ok=True)
    with open(INTERIM / "corp_codes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["corp_code", "corp_name", "corp_eng_name", "stock_code", "modify_date"])
        for el in root.iter("list"):
            w.writerow([(el.findtext(k) or "").strip() for k in
                        ["corp_code", "corp_name", "corp_eng_name", "stock_code", "modify_date"]])


def _read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def step_overview(c: DartClient):
    """company.json for every corp with a stock_code -> corp_cls (Y/K/N/E), induty_code, jurir_no."""
    rows = [r for r in _read(INTERIM / "corp_codes.csv") if r["stock_code"]]
    out = []
    for r in rows:
        j = c.get_json("company.json", {"corp_code": r["corp_code"]})
        out.append({k: j.get(k, "") for k in ["corp_code", "corp_name", "stock_name", "stock_code",
                    "corp_cls", "induty_code", "jurir_no", "bizr_no", "acc_mt", "status"]} | {"corp_code": r["corp_code"]})
    _write(INTERIM / "company_overview.csv", out)


def step_listed(c: DartClient):
    """Universe = corp_cls in {Y (KOSPI), K (KOSDAQ)} per company.json."""
    rows = [r for r in _read(INTERIM / "company_overview.csv") if r["corp_cls"] in ("Y", "K")]
    for r in rows:
        r["market"] = {"Y": "KOSPI", "K": "KOSDAQ"}[r["corp_cls"]]
    _write(INTERIM / "listed.csv", rows)


def step_invest(c: DartClient, years):
    out = []
    for y in years:
        for r in _read(INTERIM / "listed.csv"):
            j = c.get_json("otrCprInvstmntSttus.json",
                           {"corp_code": r["corp_code"], "bsns_year": str(y), "reprt_code": REPRT_ANNUAL})
            for it in j.get("list", []) or []:
                out.append({"src_corp_code": r["corp_code"], "bsns_year": y, "api_status": j.get("status"), **it})
            if not j.get("list"):
                out.append({"src_corp_code": r["corp_code"], "bsns_year": y, "api_status": j.get("status")})
    _write(INTERIM / "invest_raw_rows.csv", out)


# ---- entity resolution (rules documented in README) ----
_SUFFIX = re.compile(r"\(주\)|㈜|주식회사|\(유\)|유한회사|co\.?,?\s*ltd\.?|inc\.?|corp\.?|\s+", re.I)


def norm_name(s: str) -> str:
    return _SUFFIX.sub("", (s or "")).replace("（", "(").replace("）", ")").strip().lower()


def step_resolve(c: DartClient):
    """R1 exact corp_name; R2 normalized name (suffix/space removal) unique among listed;
    R3 normalized stock_name unique. Ambiguous/none -> unmatched list (kept, never dropped silently)."""
    listed = _read(INTERIM / "listed.csv")
    exact = {r["corp_name"]: r["corp_code"] for r in listed}
    by_norm, by_stock = {}, {}
    for r in listed:
        by_norm.setdefault(norm_name(r["corp_name"]), set()).add(r["corp_code"])
        by_stock.setdefault(norm_name(r["stock_name"]), set()).add(r["corp_code"])
    res = []
    for row in _read(INTERIM / "invest_raw_rows.csv"):
        name = (row.get("inv_prm") or "").strip()
        if not name:
            continue
        rule, tgt = "unmatched", ""
        if name in exact:
            rule, tgt = "R1_exact", exact[name]
        elif len(by_norm.get(norm_name(name), ())) == 1:
            rule, tgt = "R2_norm_corp_name", next(iter(by_norm[norm_name(name)]))
        elif len(by_stock.get(norm_name(name), ())) == 1:
            rule, tgt = "R3_norm_stock_name", next(iter(by_stock[norm_name(name)]))
        elif len(by_norm.get(norm_name(name), ())) > 1:
            rule = "ambiguous"
        res.append(row | {"tgt_corp_code": tgt, "match_rule": rule})
    _write(INTERIM / "invest_resolved.csv", res)
    _write(PROC / "unmatched_investees.csv", [r for r in res if r["match_rule"] in ("unmatched", "ambiguous")])


def step_build(c: DartClient):
    raise NotImplementedError("Phase B: build nodes.csv / edges.csv / edges_undirected.csv (see README rules)")


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step"); ap.add_argument("--year", type=int)
    ap.add_argument("--prev-year", type=int); a = ap.parse_args()
    c = DartClient(); years = [y for y in (a.year, a.prev_year) if y]
    steps = {"corpcodes": step_corpcodes, "overview": step_overview, "listed": step_listed,
             "invest": lambda c: step_invest(c, years), "resolve": step_resolve, "build": step_build}
    try:
        for s in (list(steps) if a.step == "all" else [a.step]):
            print("step", s); steps[s](c)
    except DailyLimitReached as e:
        print("STOPPED (resume tomorrow):", e); sys.exit(2)
