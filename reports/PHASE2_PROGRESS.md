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
