"""Descriptive stats only (no models). Usage: python stats.py --year 2025 [--backup 2024]
Reads data/processed/{nodes,edges,edges_undirected,build_counts,groups}.csv; writes reports/stats_<year>.json and figures."""
import argparse, csv, json
from collections import Counter
from pathlib import Path
import numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; P = ROOT / "data/processed"; R = ROOT / "reports"

def rd(n):
    p = P / n
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []

def dstats(d):
    d = np.asarray(d); n = len(d)
    return {"n": n, "mean": round(float(d.mean()), 3), "median": float(np.median(d)),
            **{f"p{q}": float(np.percentile(d, q)) for q in (20, 50, 80, 95, 99)}, "max": int(d.max()),
            **{f"share_deg{k}": round(float((d == k).mean()), 4) for k in (0, 1, 2)}}

def run(year, plot=True):
    nodes = rd("nodes.csv"); und = [e for e in rd("edges_undirected.csv") if int(e["bsns_year"]) == year]
    dire = [e for e in rd("edges.csv") if int(e["bsns_year"]) == year]
    G = nx.Graph(); G.add_nodes_from(n["corp_code"] for n in nodes); G.add_edges_from((e["u"], e["v"]) for e in und)
    cc = sorted(nx.connected_components(G), key=len, reverse=True)
    deg = dict(G.degree()); mk = {n["corp_code"]: n["market"] for n in nodes}
    nm = {n["corp_code"]: n["corp_name"] for n in nodes}
    grp = {g["corp_code"]: g for g in rd("groups.csv")}
    s = {"year": year, "nodes": G.number_of_nodes(), "edges_undirected": G.number_of_edges(), "edges_directed": len(dire),
         "both_direction_pairs": sum(e["directions"] == "both" for e in und),
         "components": len(cc), "lcc_size": len(cc[0]), "lcc_share": round(len(cc[0]) / G.number_of_nodes(), 4),
         "non_isolated_components": sum(len(c) > 1 for c in cc),
         "degree_all": dstats(list(deg.values())),
         "degree_by_market": {m: dstats([deg[c] for c in deg if mk[c] == m]) for m in ("KOSPI", "KOSDAQ")},
         "nodes_by_market": dict(Counter(mk.values())),
         "edge_ratio_missing": sum(e["ownership_pct"] in ("", None) for e in dire),
         "edges_per_node": round(G.number_of_edges() / G.number_of_nodes(), 4)}
    top = sorted(deg, key=lambda c: (-deg[c], c))[:20]
    s["top20"] = [{"corp_code": c, "corp_name": nm[c], "market": mk[c], "degree": deg[c],
                   "group_proxy": grp.get(c, {}).get("group_proxy", "未获取")} for c in top]
    if grp:
        known = [e for e in und if grp.get(e["u"], {}).get("group_proxy") and grp.get(e["v"], {}).get("group_proxy")]
        s["intra_group_edges"] = {"edges_with_both_groups_known": len(known),
                                  "same_group_proxy": sum(grp[e["u"]]["group_proxy"] == grp[e["v"]]["group_proxy"] for e in known)}
    bc = json.loads((P / "build_counts.json").read_text()).get(str(year), {})
    s["build_counts"] = bc
    tot = bc.get("rows_total", 0)
    s["row_coverage_kept"] = round(bc.get("kept_rows", 0) / tot, 4) if tot else None
    src_with_rows = {e["source_corp_code"] for e in dire}
    s["share_nodes_out_edge"] = round(len(src_with_rows) / len(nodes), 4)
    R.mkdir(exist_ok=True); (R / f"stats_{year}.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
    if plot:
        d = np.array(list(deg.values())); cnt = Counter(d)
        xs = sorted(cnt); fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar(xs, [cnt[x] for x in xs]); ax[0].set_xlabel("degree"); ax[0].set_ylabel("#nodes"); ax[0].set_title(f"Degree distribution (linear), FY{year}")
        xp = [x for x in xs if x > 0]; ax[1].loglog(xp, [cnt[x] for x in xp], "o")
        ax[1].set_xlabel("degree (log)"); ax[1].set_ylabel("#nodes (log)"); ax[1].set_title(f"Degree>0 (log-log), FY{year}")
        fig.tight_layout(); fig.savefig(R / f"degree_dist_{year}.png", dpi=120)
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in ("KOSPI", "KOSDAQ"):
            dm = np.sort([deg[c] for c in deg if mk[c] == m])[::-1]
            ax.loglog(np.arange(1, len(dm) + 1), np.maximum(dm, 0.5), label=m)
        ax.set_xlabel("rank"); ax.set_ylabel("degree (0 shown as 0.5)"); ax.legend(); ax.set_title(f"Rank-degree by market FY{year}")
        fig.tight_layout(); fig.savefig(R / f"degree_rank_by_market_{year}.png", dpi=120)
    return s

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--year", type=int, required=True); ap.add_argument("--backup", type=int)
    a = ap.parse_args(); print(json.dumps(run(a.year), ensure_ascii=False, indent=1)[:3000])
    if a.backup: run(a.backup)
