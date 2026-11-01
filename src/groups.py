"""Approximate business-group proxy (NOT the official 공정위 기업집단 designation; 未验证).
For each node with undirected degree>=1 in --year: call hyslrSttus (최대주주 및 특수관계인, 사업보고서),
take the holder with relate containing '본인' (else max trmend ratio); if that holder name resolves to a
listed company (exact / normalized name), climb to that company's controller (max depth 6).
group_proxy = name at the top of the chain. Output data/processed/groups.csv."""
import argparse, csv
from collections import Counter
from dart_client import DartClient
from pipeline import INTERIM, PROC, _read, _write, norm_name
from build import pct

def controller(c, code, year):
    j = c.get_json("hyslrSttus.json", {"corp_code": code, "bsns_year": str(year), "reprt_code": "11011"})
    rows = [r for r in (j.get("list") or []) if r.get("nm") and r["nm"].strip() not in ("계", "합계", "-")]
    if not rows:
        return None, j.get("status")
    me = [r for r in rows if "본인" in (r.get("relate") or "")]
    best = me[0] if me else max(rows, key=lambda r: pct(r.get("trmend_posesn_stock_qota_rt")) or -1)
    return best["nm"].strip(), j.get("status")

def main(year):
    c = DartClient()
    listed = _read(INTERIM / "listed.csv")
    name2code = {}
    for r in listed:
        for n in (r["corp_name"], r["stock_name"]):
            name2code.setdefault(norm_name(n), set()).add(r["corp_code"])
    und = [e for e in _read(PROC / "edges_undirected.csv") if int(e["bsns_year"]) == year]
    deg = Counter([e["u"] for e in und] + [e["v"] for e in und])
    cache, out = {}, []
    def ctrl(code):
        if code not in cache:
            cache[code] = controller(c, code, year)
        return cache[code]
    for code in sorted(deg):
        chain, cur, status = [], code, None
        name, status = ctrl(cur)
        while name:
            chain.append(name)
            nxt = name2code.get(norm_name(name), set())
            if len(nxt) != 1 or len(chain) > 6 or next(iter(nxt)) == cur:
                break
            cur = next(iter(nxt)); name2, _ = ctrl(cur)
            if not name2 or name2 in chain:
                break
            name = name2
        out.append({"corp_code": code, "direct_controller": chain[0] if chain else "",
                    "group_proxy": chain[-1] if chain else "", "chain": " > ".join(chain), "hyslr_status": status})
    _write(PROC / "groups.csv", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--year", type=int, required=True); main(ap.parse_args().year)
