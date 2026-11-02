# PHASE2_PROGRESS (append-only log; KST)

## 2026-11-01 21:40 Step 1 started
- Fixed `src/dart_client.py`: the call counter now uses a cross-process file lock (fcntl) with atomic writes. Earlier, concurrent processes undercounted the calls. Today's count was reset to the number of cache files fetched today (13,770), which is a conservative value.
- Added `src/corrections.py` (offline) → `reports/corrections_2025.json`, `data/processed/edge_pct_audit_2025.csv`.
- Added `src/filings.py` (list.json A001/I003, 2019-01~2026-10, quarterly windows, last_reprt_at=N) → `data/interim/filings_<ty>.csv`, running in the background (logs/filings.log).

## 22:10 (d) Edge disclosure dates
- filings_A001.csv: 32,236 사업보고서-type filings (2019-01 to 2026-10, originals plus corrections). For 31,892 of them, the first 8 digits of rcept_no equal rcept_dt; the 344 that differ are mostly late-night filings dated the next day.
- All 2,087 FY2025 edges matched the index: 2,048 were filed in 2026 and 39 in 2025 (non-December fiscal years); 1,894 were filed on or before 2026-04-30; **411 come from correction filings** ([기재정정] etc.).
  → otrCprInvstmntSttus returns the latest version, not the as-of-C version (limitation). → `data/processed/edges_dated.csv`, `reports/edge_dates.json`.
- Step 2 probe: fnlttMultiAcnt is available for FY2015–2025 (2014 returns 013). Fields: assets, liabilities, equity, revenue, operating income, net income, pre-tax income; 자본금 is present in some years; **no interest expense**; each row has rcept_no.
- otrCprInvstmntSttus before FY2020: the structured table is empty for some companies (삼성전자, 현대차, SK하이닉스, 삼성물산 return only "-"/"합계" for 2018–2020), while 네이버, 대한항공 and 경방 have data. Coverage is 未验证 until a full fetch.

## 22:40 Step 1 (a)–(g) done → reports/CORRECTIONS.md
- universe.py: 3,270 companies in the universe (2,610 listed now, 553 not listed now, 107 Konex); delisting_review.csv has 300 rows.
- financials.py is running in the background: fnlttMultiAcnt for FY2018–2025, about 264 calls (logs/financials.log).

## 23:00 Step 2 data availability
- fnlttMultiAcnt FY2018–2025 has been fetched (`data/interim/fin_multi_raw.csv`, 236 calls). Coverage is in `reports/fin_coverage.json`. OFS covers 2,006/2,305 (87%) in 2018, rising to 2,764/2,820 (98%) in 2025; CFS covers 70–78%.
  Decision: **OFS is the main specification** (ownership edges are parent-level investments, and OFS covers more firms); CFS is a sensitivity check. Interest expense is not in fnlttMultiAcnt; the reason for the ~300 uncovered firms per year in 2018–2022 (financial firms? no XBRL?) is 未验证.
- invest_panel.py is running in the background (logs/invest_panel.log): otrCprInvstmntSttus for FY2023→2019, then 2024 and 2025 for formerly listed firms. Estimate: about 14k calls. It stops cleanly when today's budget (18,000) runs out; rerun the same command after 00:00 KST tomorrow to resume.

## 2026-11-02 Investment panel fetch complete + sample decision
- invest_panel.py finished in one day (11,986 calls): `data/interim/invest_panel_raw.csv`, 227,810 rows, FY2019–2025. api_status 000 for 227,800 rows, 013 for 10.
- Firms per year with at least one named investee row (excluding "-"/합계/계): 2019 1,841/2,408; 2020 1,947/2,486; 2021 2,173/2,552; 2022 2,235/2,636; 2023 2,302/2,722; 2024 2,371/2,804; 2025 2,417/2,820.
- **Decision (Jack, 2026-11-02): the main panel starts at FY2021.** FY2019 and FY2020 are used only as a sensitivity check (lower structured-table coverage, see the 2018–2020 empty-table note above).

## Phase 2B Step 1 — as-of original versions (2026-11-02 KST)
- versions_index: 394 targets (314 validation FY2024 + 80 late-label train candidates, superset of baseline 14+63), 904 versions; each stored under data/raw/versions/<rcept_no>/ (document.zip + fnlttXbrl.zip, never overwritten).
- As-filed values: document ACODE cells (FY2024 docs); FY2022/23 docs lack ACONTEXT → fallback to per-rcept_no XBRL instance (CFY{fy}…SeparateMember). On 722 versions with both, doc vs XBRL agree 100% (assets/OI/NI/equity) → XBRL per rcept_no is as-filed.
- Validation: 313/314 have a version by C (01144028 씨티케이 first filed 20250530 → stays missing); 303 assets+OI recovered; 10 missing OI = financial firms (securities/bank/insurance/holding) using entity-specific income tags (未验证 mapping, left missing). As-of≠final: assets 11, liab 10, equity 14, capital 0, revenue 5, OI 7, NI 14. OI-sign label flips 0/303.
- Train candidates: 79/80 version by C (01437186 none), 78 recovered; as-of≠final: assets 25, liab 11, equity 24, OI 14, NI 27; label flip 1 (00308896 FY2023); 00923899 FY2023 has version but no OI tag (未验证).
- Parser check (latest version vs fnlttMultiAcnt final): 100% agree on all compared accounts both groups.

## Phase 2B Step 2 — isolated audit (2026-11-02 KST)
- fetch_docs: original FY2024(.12) 사업보고서 document (first non-correction filing ≤C) for 2,745 of 3,270 firms (rest: no Dec-FYE original by C). Calls today 16,298/18,000.
- Baseline FY2024 graph reproduced exactly (1,570 edges; 1,069 isolated all degree 0).
- Doc-based categories (data/processed/isolated_audit.csv): iii_external_unlisted 670, iv_filing_lists_no_relation 211, iii_listed_unmatched_未验证 75, ii_name_match_failed 58, i_api_table_incomplete 55 (mostly API returns a correction filed after C), i_doc_not_obtained 0.
- 361 universe edges in as-filed originals but not in baseline graph (data/processed/doc_recovered_edges_2024.csv): 242 API row belongs to a correction filed after C (baseline timing filter dropped a relation already in the original); 97 investee named by its as-filed former name (verified rename chain); 22 absent from API table; 3 no API rows; 1 other. 113 isolated firms would gain ≥1 edge. NOT added to any graph.
- 53 no-financial-neighbour firms: 56 neighbour links; 24 now have as-of assets (document or XBRL of the original). Rest: 12 neighbours non-December FYE (no 2024.12 original), ~17 non-XBRL filers (fnlttXbrl status 014, untagged tables not parsed), 3 financial firms (entity-specific tags).
