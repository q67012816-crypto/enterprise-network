"""Phase 2 Step 1 corrections, offline (no API calls). Usage: python corrections.py --year 2025
Writes reports/corrections_<year>.json and data/processed/edge_pct_audit_<year>.csv."""
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path
import networkx as nx
from build import pct
ROOT = Path(__file__).resolve().parents[1]; P = ROOT / "data/processed"; I = ROOT / "data/interim"; R = ROOT / "reports"
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
NONCLEAN = {"의견거절", "한정의견", "부적정의견"}

def num(s):
    v = pct(s); return v

def classify_rows(rows):
    """rows: raw otrCprInvstmntSttus rows behind one edge. Returns class for zero/missing pct edges."""
    qy = [num(r["trmend_blce_qy"]) for r in rows]; bk = [num(r["trmend_blce_acntbk_amount"]) for r in rows]
    if any(q and q > 0 for q in qy) or any(b and b > 0 for b in bk):
        return "holding_present_pct_zero_or_missing (rounding/missing pct)"
    if all(q in (None,) for q in qy) and all(b is None for b in bk):
        return "all_period_end_fields_missing"
    return "period_end_zero (exited/disposed)"

def main(year):
    out = {}
    nodes = rd(P / "nodes.csv"); und = [e for e in rd(P / "edges_undirected.csv") if int(e["bsns_year"]) == year]
    dire = [e for e in rd(P / "edges.csv") if int(e["bsns_year"]) == year]
    G = nx.Graph(); G.add_nodes_from(n["corp_code"] for n in nodes); G.add_edges_from((e["u"], e["v"]) for e in und)
    deg = dict(G.degree()); N, E = G.number_of_nodes(), G.number_of_edges()
    out["degree_definition"] = {"nodes": N, "undirected_edges": E, "mean_degree_2E_over_N": round(2 * E / N, 3),
                                "edges_per_node_E_over_N": round(E / N, 3), "isolated": sum(d == 0 for d in deg.values())}
    # (a) audit label denominators
    import re
    aud = rd(P / "labels_audit_opinion.csv"); per = defaultdict(set); per_strict = defaultdict(set)
    kmax = defaultdict(int)
    for r in aud:
        m = re.search(r"제\s*(\d+)\s*기", r["period"])
        if m: kmax[r["corp_code"]] = max(kmax[r["corp_code"]], int(m.group(1)))
    for r in aud:
        op = r["adt_opinion"].strip()
        if "(당기)" in r["period"]:
            per_strict[r["corp_code"]].add(op)
        m = re.search(r"제\s*(\d+)\s*기", r["period"])
        if (m and int(m.group(1)) == kmax[r["corp_code"]]) or (not m and "(당기)" in r["period"]):
            per[r["corp_code"]].add(op)
    vs = {c for c, s_ in per_strict.items() if s_ - {""}}
    out["audit_label_strict_dangi_string_only"] = {"valid": len(vs), "positives": sum(bool(per_strict[c] & NONCLEAN) for c in vs),
        "deg0_valid": sum(deg.get(c, -1) == 0 for c in vs), "deg0_pos": sum(deg.get(c, -1) == 0 and bool(per_strict[c] & NONCLEAN) for c in vs)}
    valid = {c: s - {""} for c, s in per.items() if s - {""}}
    pos = {c for c, s in valid.items() if s & NONCLEAN}
    by = {}
    for name, f in (("deg0", lambda d: d == 0), ("deg1_2", lambda d: 1 <= d <= 2), ("deg3plus", lambda d: d >= 3)):
        cs = [c for c in valid if c in deg and f(deg[c])]
        by[name] = {"valid": len(cs), "positives": sum(c in pos for c in cs)}
    out["audit_label"] = {"rule": "당기 = rows with the highest 제N기 per company (fallback: period contains (당기)), since ~106 companies format period without (당기); blank opinion excluded; positive if any row in 의견거절/한정의견/부적정의견",
                          "companies_with_any_row": len(per), "valid_nonblank": len(valid), "positives": len(pos),
                          "valid_in_node_set": sum(c in deg for c in valid), "by_degree": by}
    # (c) zero / missing pct
    res = [r for r in rd(I / "invest_resolved.csv") if r["bsns_year"] == str(year) and r.get("tgt_corp_code")]
    rows_by = defaultdict(list)
    for r in res:
        rows_by[(r["src_corp_code"], r["tgt_corp_code"])].append(r)
    audit, cls = [], Counter()
    for e in dire:
        p_ = e["ownership_pct"]; kind = "missing" if p_ in ("", None) else ("zero" if float(p_) == 0 else None)
        if not kind:
            continue
        rs = rows_by.get((e["source_corp_code"], e["target_corp_code"]), [])
        c = classify_rows(rs) if rs else "raw_rows_not_found"
        cls[(kind, c)] += 1
        audit.append({"source": e["source_corp_code"], "target": e["target_corp_code"], "kind": kind, "class": c,
                      "trmend_qy": "|".join(r["trmend_blce_qy"] for r in rs), "trmend_book": "|".join(r["trmend_blce_acntbk_amount"] for r in rs),
                      "bsis_pct": "|".join(r["bsis_blce_qota_rt"] for r in rs), "inv_prm": e["inv_prm"]})
    with open(P / f"edge_pct_audit_{year}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(audit[0])); w.writeheader(); w.writerows(audit)
    out["zero_missing_pct"] = {f"{k}|{c}": v for (k, c), v in sorted(cls.items())}
    def iso(drop):
        H = nx.Graph(); H.add_nodes_from(G.nodes)
        dropped = {(a["source"], a["target"]) for a in audit if drop(a)}
        for e in dire:
            if (e["source_corp_code"], e["target_corp_code"]) not in dropped:
                H.add_edge(e["source_corp_code"], e["target_corp_code"])
        H.remove_edges_from(nx.selfloop_edges(H))
        return {"edges": H.number_of_edges(), "isolated": sum(d == 0 for _, d in H.degree())}
    out["isolated_sensitivity"] = {"baseline": iso(lambda a: False), "drop_zero_pct": iso(lambda a: a["kind"] == "zero"),
                                   "drop_exited_only": iso(lambda a: "exited" in a["class"]),
                                   "drop_zero_and_missing": iso(lambda a: True)}
    # (e) ownership bands (presumption labels only)
    bands = Counter()
    for e in dire:
        v = pct(e["ownership_pct"])
        bands["missing" if v is None else "0" if v == 0 else "(0,5)" if v < 5 else "[5,20)" if v < 20 else "[20,50] significant-influence presumption (K-IFRS 1028), NOT control" if v <= 50 else ">50 majority-owned (control likely, 未验证 per entity)"] += 1
    out["ownership_bands"] = dict(bands)
    # (d) disclosure dates from rcept_no prefix
    out["rcept_no_year"] = dict(Counter(e["rcept_no"][:4] for e in dire))
    # (g) group label coverage
    grp = rd(P / "groups.csv")
    out["group_labels"] = {"rows": len(grp), "with_proxy": sum(bool(g["group_proxy"]) for g in grp),
                           "non_isolated_nodes": sum(d > 0 for d in deg.values()), "isolated_nodes_with_label": 0,
                           "note": "labels only computed for degree>=1 nodes; not used to add edges"}
    R.mkdir(exist_ok=True); (R / f"corrections_{year}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--year", type=int, default=2025); main(ap.parse_args().year)
