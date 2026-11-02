# Phase 2B report — as-of versions & isolated-firm audit (2026-11-02 KST)

**Caveat:** the test year (FY2024 features → 2025 labels) was already viewed in the baseline. Any re-run on it is **exploratory, not confirmatory**.
Nothing in this phase changes the graph or retrains a model; recovered edges/values are written as separate tables only.

## Step 1 — original (as-filed) versions
| group | targets | version filed ≤C | assets+OI recovered | as-of ≠ final (assets/liab/equity/capital/rev/OI/NI) | OI-sign label flips |
|---|---|---|---|---|---|
| validation FY2024, C=20250430 | 314 | 313 | 303 | 11/10/14/0/5/7/14 | 0/303 |
| late-label train (FY2022/23, C=20240430; superset of baseline 14+63) | 80 | 79 | 78 | 25/11/24/0/5/14/27 | 1/78 (00308896 FY2023) |

- 904 versions stored under `data/raw/versions/<rcept_no>/` (document.zip + fnlttXbrl.zip; never overwritten; `.part`→rename).
- Values: document ACODE cells (FY2024 docs); pre-2024 docs lack ACONTEXT → the per-rcept_no XBRL instance (context `CFY{fy}…SeparateMember`). On 722 versions with both sources, they agree 100% → fnlttXbrl per rcept_no is as-filed.
- Parser check: the latest version parsed vs fnlttMultiAcnt final values agree 100% on every compared account.
- Not recovered: 01144028 씨티케이 (first filing 20250530 > C, kept missing — correct as-of); 01437186 (no FY filing); 10 validation financial firms (securities/bank/insurance/holding) + 00923899 FY2023: operating income uses entity-specific tags, not mapped (未验证).

## Step 2 — 1,069 isolated test firms
Baseline graph reproduced exactly (1,570 edges). Categories using the original FY2024 document filed ≤C:

| category | firms |
|---|---|
| iii investees all outside universe / unlisted (kept as external, not added) | 670 |
| iv filing's 타법인출자 table lists no relation (해당사항 없음 / only '-' / 합계) | 211 |
| iii listed investee not matched (foreign/KONEX/delisted or name variant) — 未验证 | 75 |
| ii name→corp_code failed; fixed by effective-dated as-filed name | 58 |
| i API table gap (mostly the API row belongs to a correction filed after C) | 55 |
| i document not obtained / parse failed | 0 |

Rules for (ii): names are matched exact, then suffix-normalised, then by stock name, accepting a match only when it is unique. Former names come from the firm's own filed COMPANY-NAME, chained through its own "상호의 변경" table. Only filings ≤C count. A former name that equals another firm's current name is rejected. No fuzzy matching.

**Key finding:** 361 universe-internal relations appear in as-filed originals but are missing from the baseline graph (+23% vs 1,570):
- 242: the API returns the row of the latest correction (filed after C), so the baseline timing filter dropped a relation that already existed in the original filed ≤C.
- 97: the investee is named by its as-filed former name; the baseline used current names.
- 22: absent from the API table; 3 have no API rows; 1 other.
- 113 isolated firms would gain ≥1 edge.
- File: `data/processed/doc_recovered_edges_2024.csv`, column `reason`. Not merged into any graph.

Former-name review: all 80 unique mappings were read by eye and they match known renames (e.g. 포스코→포스코홀딩스, DGB금융지주→아이엠금융지주, 엔씨소프트→엔씨). A formal check is still 未验证.

**53 connected firms with no financial neighbour:** 56 neighbour links; 24 now have as-of assets. Remaining:
- 12 non-December FYE neighbours (no 2024.12 original);
- about 17 non-XBRL filers (fnlttXbrl status 014; untagged tables, not parsed);
- 3 financial firms.
File: `data/processed/nofin_recheck.csv`.

## Step 3 — not started (prep only, next dispatch)

## Limitations
- Financial-firm OI tags.
- Non-December FYE excluded.
- Untagged (non-XBRL) statements not parsed.
- (iii-listed) 75 firms 未验证.
- The doc invest-table parser takes the first table with INV_PRM rows.

## Reproduce (from src/, OPENDART_API_KEY set in env)
```
python versions.py && python fetch_docs.py      # downloads (budget-guarded, resumable)
python parse_versions.py && python asof.py      # Step 1 → data/processed/asof_financials.csv, reports/asof_recovery.json
python isolated_audit.py && python isolated_audit_doc.py && python nofin_recheck.py   # Step 2
```
