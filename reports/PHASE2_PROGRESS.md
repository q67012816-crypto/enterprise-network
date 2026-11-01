# PHASE2_PROGRESS (append-only log; KST)

## 2026-11-01 21:40 Step 1 started
- Fixed `src/dart_client.py`: the call counter now uses a cross-process file lock (fcntl) with atomic writes. Earlier, concurrent processes undercounted the calls. Today's count was reset to the number of cache files fetched today (13,770), which is a conservative value.
- Added `src/corrections.py` (offline) → `reports/corrections_2025.json`, `data/processed/edge_pct_audit_2025.csv`.
- Added `src/filings.py` (list.json A001/I003, 2019-01~2026-10, quarterly windows, last_reprt_at=N) → `data/interim/filings_<ty>.csv`, running in the background (logs/filings.log).
