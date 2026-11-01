"""Coverage of fnlttMultiAcnt by FY: firms requested vs returned, OFS/CFS, key accounts. -> reports/fin_coverage.json"""
import csv, json
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
U = [r for r in rd(ROOT / "data/processed/universe_firms.csv") if r["status_now"] != "konex_now"]
F = rd(ROOT / "data/interim/fin_multi_raw.csv")
KEY = ["자산총계", "부채총계", "자본총계", "자본금", "매출액", "영업이익", "당기순이익(손실)"]
have = defaultdict(lambda: defaultdict(set))
for r in F:
    have[(r["bsns_year"], r["fs_div"])][r["account_nm"]].add(r["corp_code"])
out = {}
for y in range(2018, 2026):
    req = {r["corp_code"] for r in U if str(y) in r["fy_with_annual_report"].split()}
    d = {"requested": len(req)}
    for fs in ("OFS", "CFS"):
        a = have[(str(y), fs)]; anyc = set().union(*a.values()) if a else set()
        d[fs] = {"firms_any": len(anyc & req), **{k: len(a.get(k, set()) & req) for k in KEY}}
    out[y] = d
(ROOT / "reports/fin_coverage.json").write_text(json.dumps(out, ensure_ascii=False, indent=1)); print(json.dumps(out, ensure_ascii=False))
