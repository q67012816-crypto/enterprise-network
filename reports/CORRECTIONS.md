# CORRECTIONS (Phase 2 Step 1, 2026-11-01)

These are corrections to the Phase 1 reports (DEGREE_REPORT / LABEL_FEASIBILITY) and the summary. Each number below can be reproduced offline:
`cd src && ../.venv/bin/python corrections.py --year 2025 && ../.venv/bin/python edge_dates.py && ../.venv/bin/python universe.py`
Outputs: `reports/corrections_2025.json`, `reports/edge_dates.json`, `reports/universe.json`, `data/processed/{edge_pct_audit_2025,edges_dated,universe_firms,delisting_review}.csv`.

## (a) Audit-label denominator
- Rule: blank opinions are excluded from the denominator. A company is positive if any of its 당기 rows is 의견거절, 한정의견 or 부적정의견.
- **The reviewer's 2,482 / 1,068 could not be reproduced exactly from the saved data.** Two definitions give two results:
  - Strict definition (period string contains "(당기)", the same rule as Phase 1): **2,481** companies with a valid opinion, 38 positives. Degree-0 group: **1,067** companies, 26 positives.
  - Corrected definition: 당기 = the row with the largest 제N기 number for each company. This is needed because about 106 companies write the period as "제25기" or "(2025년)" without "(당기)". Result: **2,539** companies with a valid opinion, **40** positives. Degree 0: 1,088 companies, 26 positives. Degree 1–2: 983 / 12. Degree ≥3: 468 / 2.
  - Phase 1's "38 of 2,481" was correct only under the strict definition, and the strict definition misses about 58 companies. From now on the corrected definition is used. The difference from the reviewer's numbers (±1) is 未验证.
  - Positive rate by group: degree 0 = 2.4%, degree ≥1 = 1.0%. Isolated companies have the higher rate.

## (b) Mean degree
- Mean degree = 2E/N = 2×2,002/2,655 = **1.508**. Edges per node E/N = **0.754**. Phase 1's "0.75 edges per node" is E/N; do not read it as the mean degree.

## (c) Edges with 0% or missing ownership (FY2025, directed, 2,087 edges)
Classified from the raw otrCprInvstmntSttus rows (period-end shares trmend_blce_qy and book value trmend_blce_acntbk_amount). Details are in `edge_pct_audit_2025.csv`.

| Type | Shares or book value still held at period end (rounding / % not filled in) | Period-end shares and book value are 0 (exited) | All period-end fields empty |
|---|---|---|---|
| pct = 0 (241) | 183 | 41 | 17 |
| pct missing (317) | 159 | 5 | 153 |

Sensitivity of the isolated count (undirected):

| Scenario | Edges | Isolated |
|---|---|---|
| Baseline | 2,002 | 1,154 |
| Drop pct=0 edges | 1,763 | **1,203** |
| Drop only exited edges (46) | 1,959 | 1,168 |
| Drop pct=0 and missing | 1,454 | 1,284 |

Recommendation: drop the exited edges (period-end value 0). Keep "holding present but % is 0 or missing" edges with a flag. Treat "all fields empty" as 未验证.

## (d) Edge disclosure dates
- Each edge's rcept_no was matched to the list.json A001 index (`filings_A001.csv`, 32,236 filings): **2,087/2,087 matched**. For 2,081 edges, rcept_dt equals the first 8 digits of rcept_no.
- FY2025: **2,048 filed in 2026**, 39 in 2025 (non-December fiscal years). **1,894** were filed on or before 2026-04-30 and 193 after. **411** come from correction filings ([기재정정]/[첨부정정]/[첨부추가]).
- FY2024: 1,904 edges; 1,608 filed on or before 2025-04-30; 561 from correction filings.
- **Limitation**: otrCprInvstmntSttus returns the current final version, so the version that was visible at date C cannot be reconstructed. An edge from a correction filed after C may be leaked information. Mitigation: use only edges whose disclosure_date ≤ C, and run a sensitivity test that drops edges from correction filings. Edge table: `data/processed/edges_dated.csv` (disclosure_date, report_nm).

## (e) Meaning of ownership-share bands (FY2025 directed edges)
- (0,5)%: 777. [5,20)%: 266. **[20,50]%: 359 — significant influence is only presumed (K-IFRS 1028); this is not control.** >50%: 127 — majority-owned; control is likely but 未验证 per entity. 0%: 241. Missing: 317.
- DEGREE_REPORT's phrase "≥20% = control or group tie" is corrected to "≥20% = presumed significant influence".

## (f) Historical listing universe and survivor bias
- Universe rule: the company has a stock_code in corpCode **and** filed an A001 사업보고서 for some fiscal year from 2018 to 2025. This gives **3,270** companies: 2,610 currently listed, 553 not listed now (corp_cls E), and 107 currently on Konex (excluded).
- Firm-years with an annual report, for listed or formerly listed companies: 2018: 2,305 · 2019: 2,409 · 2020: 2,488 · 2021: 2,556 · 2022: 2,637 · 2023: 2,724 · 2024: 2,804 · 2025: 2,820.
- OpenDART has no exact listing or delisting dates (未验证). The status at a prediction date is approximated by "annual report for FY t exists" plus the dates of I003 delisting notices.
- Delisting-keyword review (`delisting_review.csv`, classified from I003 2019-01 to 2026-10 report_nm; 300 companies not listed now):
  - cause_delisting_completed (정리매매 or 상장폐지 결정 notice found): 180
  - cause_notice_only (only "cause arose" or "risk of" notices; outcome 未验证): 101
  - Konex: 19
- Not listed now and no keyword notice: 272. The reason (merger, voluntary delisting, transfer, etc.) is 未验证, so these **are not labelled as delisting positives**.
- **Counts differ from the reviewer's**: the reviewer had 111/84; my Phase 1 labels_events.csv (2024–2025) gives 146 companies with delisting keywords, 113 of them not listed now. The windows and keywords differ, so this is 未验证.
- Warning: 211 companies are not listed now but filed an annual report for FY2025. They may have been delisted in 2026 or have a special corp_cls; 未验证.

## (g) Group labels
- groups.csv covers only the **1,501** companies with degree ≥1 (status 000: 1,487; 013: 14). There are no labels for the 1,154 isolated companies. Group labels are not used to add edges. The "29.6% same-group" figure applies only to edges whose two ends both have a label.
