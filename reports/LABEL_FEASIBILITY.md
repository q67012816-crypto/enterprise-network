> **2026-11-01 correction**: under the corrected 당기 definition, audit opinions give 40 positives out of 2,539 valid companies (degree 0: 26/1,088). See `reports/CORRECTIONS.md` (a).

# LABEL_FEASIBILITY: feasibility of candidate node labels (FY2025; no label is chosen here)

Node set: the 2,655 currently listed companies. Data comes from `reports/labels_2025.json` and `data/processed/labels_*.csv`. Script: `cd src && ../.venv/bin/python labels.py --year 2025`.
"Positives" are counted among the current nodes. Imbalance = positives / 2,655, or / the number of companies with data where stated.

| Candidate label | Source endpoint | Years available (measured) | Positives | Share / imbalance | Notes |
|---|---|---|---|---|---|
| Non-clean audit opinion (한정/부적정/의견거절), 당기 | DS002 accnutAdtorNmNdAdtOpinion (2020009), 11011 | One call returns FY2025 당기 / 전기 / 전전기 | 당기 38 (of 2,481 with an opinion) · 전기 16 · 전전기 9 | 1.5% (≈1:64) | status 013 (no data) for 50 companies. No 부적정 seen. Some companies return several rows per period (consolidated/separate etc.); a company counts as positive if any row is non-clean |
| Capital impairment ≥ partial (자본총계 < 자본금), OFS | DS003 fnlttMultiAcnt | FY2025 | 78 (of 2,576 with data) | 3.0% | ≥50% impairment: 24 (0.9%); full impairment: 11 (0.4%). CFS: 58 / 19 / 10 of 2,078 |
| 관리종목 지정 | list.json pblntf_ty=I003 + report_nm regex | calendar years 2024 and 2025 | 2024: 11 · 2025: 21 | 0.8% | Keyword matching; recall not checked by hand (未验证) |
| Listing eligibility review (상장적격성 실질심사) | list.json I003 | 2024 and 2025 | 26 · 38 | 1.4% | Same as above |
| Delisting cause / decision | list.json I003 | 2024 and 2025 | 14 · 27 | 1.0% | **Survivor bias**: companies already delisted are not nodes, so positives are undercounted |
| Trading halt (매매거래정지) | list.json I003 | 2024 and 2025 | 209 · 213 | 8.0% | Includes many technical halts (reorganizations, mergers, etc.); noisy |
| Business suspension (영업정지) | list.json B001 (DS005 bsnSp) | 2024 and 2025 | 21 · 24 | 0.9% | |
| Rehabilitation filing (회생절차개시신청) | list.json B001 (DS005 ctrcvsBgrq) | 2024 and 2025 | 4 · 10 | 0.4% | Very few positives |
| Default (부도발생) | list.json B001 (DS005 dfOcr) | 2024 and 2025 | 1 · 2 | <0.1% | Too few to use as a standalone label |
| Dissolution (해산사유발생) | list.json B001 (DS005 dsRsOcr) | 2024 and 2025 | not tallied (no hits in the summary) | — | 未验证 |
| Industry (induty_code) | DS001 company.json | current snapshot | multi-class, no positive/negative | — | Available, but it is static attributes, not risk |
| Market (KOSPI/KOSDAQ) | company.json corp_cls | current | 829 / 1,826 | 31% / 69% | Strongly correlated with degree (KOSPI average degree 2.7 vs KOSDAQ 1.0) |

## Notes
- All event labels come from keyword matching on report_nm in disclosure search (list.json; ≤3-month windows, done by quarter). Precision and recall were not checked by hand (未验证).
- The label year (calendar 2025) and the graph year (FY2025 사업보고서, filed around March 2026) can overlap. To avoid leakage, a downstream setup should use FY t graph → events in year t+1. Year t+1 (2026) events have **not been fetched** (未验证).
- Every risk label has positives ≤3% except trading halts. Combined with the 43.5% isolated nodes, the positives in degree-0 nodes will be very few. Positive counts by degree bucket have not been computed (未验证; doable offline from the CSVs as a next step).
