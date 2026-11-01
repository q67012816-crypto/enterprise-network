"""Build nodes.csv / edges.csv / edges_undirected.csv / discarded_investees.csv from data/interim.
Deterministic; rerun anytime: python build.py --year 2025 [--year 2024]"""
from __future__ import annotations
import argparse, csv, re, json
from collections import defaultdict
from pathlib import Path
from pipeline import INTERIM, PROC, _read, _write, norm_name

FOREIGN_PAT = re.compile(
    r"[A-Za-z]{3,}|有限|公司|株式会社|\b(ltd|inc|llc|gmbh|corp|limited|pte|s\.a|b\.v|l\.p)\b|"
    r"중국|미국|베트남|인도|일본|홍콩|싱가포르|유럽|독일|폴란드|멕시코|브라질|인도네시아|태국|헝가리|영국|체코|러시아|캐나다|호주|"
    r"america|china|vietnam|japan", re.I)
SUMMARY_PAT = re.compile(r"^(합\s*계|소\s*계|총\s*계|계|기타.*|-|해당\s*사항\s*없음|없음)$")


def pct(s):
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return None


def classify(name, nonlisted_norms):
    n = (name or "").strip()
    if not n or SUMMARY_PAT.match(n):
        return "summary_or_empty"
    if norm_name(n) in nonlisted_norms:
        return "non_listed_domestic"
    if FOREIGN_PAT.search(n):
        return "foreign_likely"
    return "unmatched"


def main(years):
    listed = _read(INTERIM / "listed.csv")
    lcodes = {r["corp_code"] for r in listed}
    allc = _read(INTERIM / "corp_codes.csv")
    nonlisted_norms = {norm_name(r["corp_name"]) for r in allc if r["corp_code"] not in lcodes}
    res = [r for r in _read(INTERIM / "invest_resolved.csv") if int(r["bsns_year"]) in years]
    edges, disc, cnt = {}, [], defaultdict(lambda: defaultdict(int))
    for r in res:
        y = int(r["bsns_year"]); cnt[y]["rows_total"] += 1
        tgt = r["tgt_corp_code"]
        if tgt and tgt in lcodes:
            if tgt == r["src_corp_code"]:
                cnt[y]["self_loop"] += 1; continue
            cnt[y]["kept_rows"] += 1
            k = (r["src_corp_code"], tgt, y)
            p = pct(r.get("trmend_blce_qota_rt"))
            e = edges.setdefault(k, {"source_corp_code": k[0], "target_corp_code": tgt, "ownership_pct": None,
                                     "source_api": "otrCprInvstmntSttus", "bsns_year": y, "reprt_code": "11011",
                                     "rcept_no": r.get("rcept_no"), "match_rule": r["match_rule"],
                                     "inv_prm": r["inv_prm"], "n_rows": 0})
            e["n_rows"] += 1
            if p is not None and (e["ownership_pct"] is None or p > e["ownership_pct"]):
                e["ownership_pct"] = p
        else:
            c = classify(r["inv_prm"], nonlisted_norms) if r["match_rule"] != "ambiguous" else "ambiguous"
            cnt[y]["disc_" + c] += 1
            disc.append({"bsns_year": y, "src_corp_code": r["src_corp_code"], "inv_prm": r["inv_prm"],
                         "category": c, "trmend_blce_qota_rt": r.get("trmend_blce_qota_rt")})
    E = sorted(edges.values(), key=lambda e: (e["bsns_year"], e["source_corp_code"], e["target_corp_code"]))
    _write(PROC / "edges.csv", E)
    _write(PROC / "discarded_investees.csv", disc)
    und = {}
    for e in E:
        u, v = sorted([e["source_corp_code"], e["target_corp_code"]])
        d = "uv" if e["source_corp_code"] == u else "vu"
        k = (u, v, e["bsns_year"])
        x = und.setdefault(k, {"u": u, "v": v, "bsns_year": e["bsns_year"], "weight_max": e["ownership_pct"], "dirs": set()})
        x["dirs"].add(d)
        if e["ownership_pct"] is not None and (x["weight_max"] is None or e["ownership_pct"] > x["weight_max"]):
            x["weight_max"] = e["ownership_pct"]
    U = []
    for x in und.values():
        x["directions"] = "both" if len(x["dirs"]) == 2 else next(iter(x["dirs"])); del x["dirs"]; U.append(x)
    _write(PROC / "edges_undirected.csv", sorted(U, key=lambda x: (x["bsns_year"], x["u"], x["v"])))
    nodes = [{"corp_code": r["corp_code"], "stock_code": r["stock_code"], "corp_name": r["corp_name"],
              "market": r["market"], "induty_code": r["induty_code"]} for r in listed]
    _write(PROC / "nodes.csv", sorted(nodes, key=lambda n: n["corp_code"]))
    (PROC / "build_counts.json").write_text(json.dumps({y: dict(v) for y, v in cnt.items()}, indent=1, ensure_ascii=False))
    print(json.dumps({y: dict(v) for y, v in cnt.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--year", type=int, action="append", required=True)
    main(ap.parse_args().year)
