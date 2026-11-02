# Phase 2B rerun: baseline with repaired shareholding edges

**Scope.** This is exploratory and shows association only. The FY2024→FY2025 test year had already been viewed in the baseline and in Phase 2B, so these results are not a new untouched holdout and say nothing causal.

The sample, financial features, labels, split, the 314 validation exclusions, the hyperparameters, the threshold rule (validation max-F1 on 2,071 rows, largest tie) and the bootstrap scheme (500 reps, seed 20261102) are all fixed as in the baseline. Only the graph changes. No GNN was used and no API calls were made.

**Code.** Everything is in `reruns/repaired-edges/`:
- `src/fin_panel.py`
- `src/build_panel.py VARIANT OUTDIR`
- `src/train_eval.py OUTDIR`, run in `.venv39`: Python 3.9 with sklearn 1.6.1, numpy 1.26.4 and pandas 2.3.3, matching the baseline environment.
- `src/compare.py`

## 0. Reconciliations (done before the rerun)

### (a) 365 recovered rows vs 361 edges
The 361 figure is the deduplicated count.

| category (reason) | rows | unique directed (src→target) pairs |
|---|---|---|
| A: API row exists only in a correction filed after C | 242 | 238 |
| B: name as filed ≠ current name (verified former name) | 97 | 97 |
| C/D/E: other filter 1, no API rows 3, name absent from API table 22 | 26 | 26 |
| **total** | **365** | **361** |

- **Duplicates.** All 4 duplicates are within category A, from one filing (00110893, rcept 20250313001301). That filing lists the same target twice, for 4 targets (01275489, 01685251, 01703849, 01706794), with different share lines.
- **Cross-category overlap:** 0 pairs.
- **Self-loops:** 0.
- **Already present in the baseline 2024 graph (same direction):** 0.
- **Direction:**
  - 20 recovered pairs are the reverse of an existing baseline edge. They are kept, because the graph is directed, but they add no new undirected neighbor.
  - 1 reciprocal pair exists inside the recovered set (00102858↔00356361).
  - That gives 360 unique undirected pairs.
- **Graph check:** all 361 pairs have both endpoints observed by C=20250430, and all are added. The 2024 directed edge count goes from 1,570 to 1,931.

### (b) 80 late-label training candidates vs the baseline's 77
- My recomputed dropped list is 14 (feature FY2021) + 63 (FY2022) = 77. It matches the baseline sample-flow-detail.
- All 77 are among the 80 candidates. In the candidates file, `fy` is the label year: 15 for 2022 and 65 for 2023.
- The 3 extras never reached the baseline label-date filter:

| corp_code | label FY | why not in the baseline's eligible set |
|---|---|---|
| 01098792 | 2022 (feature 2021) | No FY2021 investment-in-other-corporations disclosure, so the firm is not "source-observed" and fails the baseline eligibility rule before the label-date check. |
| 01416235 | 2023 (feature 2022) | Reports in USD (HKD before that); the baseline keeps KRW only. |
| 01437186 | 2023 (feature 2022) | November fiscal year end (asset period 1130); the baseline needs a 1231 year end. It also has no versions in the index. |

### Panel rebuild fix
- The baseline's financial audit panel was not delivered, so I rebuilt it from `fin_multi_raw`.
- At first the rebuild added one firm the baseline lacks: 00161444, a bank. It has conflicting 영업이익 and 영업이익(손실) values.
- Fix: when an account has conflicting duplicate values, it is set to missing. With that rule the rebuilt original panel equals the baseline `model-panel.csv` exactly (7,871 rows, 0 mismatched cells, neighbor means included).
- Splits are train 3,607 / validation 2,071 / test 2,193.

## 1. Reproduction check (original graph)
I retrained on the rebuilt original panel and compared against the baseline:
- `metrics.csv` and `group-metrics.csv`: maximum absolute difference 1.2e-12 (thresholds and confusion counts included).
- `paired-bootstrap.csv`: identical.
- Test and validation probabilities: maximum difference 3.5e-11.

**The reproduction passes.**

## 2. Graph variants
| variant | definition | train/val graphs | test graph |
|---|---|---|---|
| original | baseline | baseline | baseline |
| **r2024doc** | adds the 361 doc-verified edges to FY2024 only | unchanged | repaired |
| **rall_api** | same API-only rules in every year: (A) an API row from a correction filed after C is kept if the original annual report was filed by C; (B) a uniquely verified former name with an as-of name-history check | repaired | repaired |

- **Why 2021–2023 can't get the document-verified repair:** it would need the original 2021–2023 documents, about 8k calls, which is more than the remaining budget. So r2024doc builds the training/validation graphs and the test graph differently.
- **Consequence for r2024doc:** the models learn degree and neighbor weights from the sparser, unrepaired graphs and are then applied to a denser test graph. That is a covariate shift in the degree and neighbor features.
- **How rall_api avoids this:** it uses the same rules in every year. As a check on its proxy, its 2024 additions (356) overlap the doc-verified set in 354 pairs. There are 2 API-only pairs and 7 doc-only pairs (the doc-only ones are mostly category E/D, which the API cannot see).

| | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|
| edges added, r2024doc | 0 | 0 | 0 | 361 |
| edges added, rall_api | 275 | 155 | 116 | 356 |
| nodes whose degree changed (all observed nodes), r2024doc | 0 | 0 | 0 | 423 |
| nodes whose degree changed, rall_api | 333 | 191 | 141 | 416 |
| isolated observed nodes: original / r2024doc / rall_api | 1579/1579/1393 | 1480/1480/1387 | 1441/1441/1383 | 1466/1272/1275 |

Model-sample rows (isolated = degree 0):

| split | n | isolated: original → r2024doc / rall_api | rows whose degree changed: r2024doc / rall_api | connected but no neighbor financials: original → r2024doc / rall_api |
|---|---|---|---|---|
| train | 3607 | 1909 → 1909 / 1809 | 0 / 223 | 181 → 181 / 281 |
| validation | 2071 | 1015 → 1015 / 991 | 0 / 75 | 35 → 35 / 59 |
| test | 2193 | **1069 → 956 / 955** | 270 / 266 | **53 → 114 / 116** |

The "no-neighbor-financials" count grows. Many new neighbors are targets with no eligible financial row: non-KRW or non-December filers, financial firms, or firms outside the sample. For these firms the neighbor-mean features are missing and are imputed.

## 3. Test metrics (2,193 rows; validation-chosen thresholds)
| family | features | AP orig / r2024doc / rall_api | ROC-AUC orig / r2024doc / rall_api | Brier orig / r2024doc / rall_api |
|---|---|---|---|---|
| logistic | financial | .8215 / .8215 / .8215 | .8868 / .8868 / .8868 | .1247 / .1247 / .1247 |
| logistic | +degree | .8215 / .8217 / .8214 | .8865 / .8866 / .8864 | .1248 / .1249 / .1250 |
| logistic | +degree+neighbors | .8220 / .8196 / .8191 | .8859 / .8847 / .8845 | .1253 / .1257 / .1258 |
| HGB | financial | .8360 / .8360 / .8360 | .8939 / .8939 / .8939 | .1197 / .1197 / .1197 |
| HGB | +degree | .8361 / .8355 / .8364 | .8934 / .8933 / .8930 | .1199 / .1201 / .1199 |
| HGB | +degree+neighbors | .8315 / .8297 / .8289 | .8906 / .8911 / .8904 | .1208 / .1207 / .1209 |

The financial-only models are identical by construction, which serves as a sanity check.

## 4. Paired bootstrap of the AP difference (500 reps, seed 20261102, test firms)
Neighbors minus financial, within each variant:

| family | original | r2024doc | rall_api |
|---|---|---|---|
| logistic | +.0005 [−.0043, .0046] | −.0019 [−.0079, .0031] | −.0024 [−.0083, .0028] |
| HGB | −.0045 [−.0113, .0028] | −.0063 [−.0143, .0016] | −.0071 [−.0140, .0012] |

Neighbors minus degree:
- Logistic: −.0021 [−.0074, .0029] for r2024doc and −.0023 [−.0077, .0028] for rall_api.
- HGB: −.0058 [−.0130, .0014] for r2024doc and **−.0075 [−.0144, −.0003]** for rall_api. This is the only interval that excludes 0, and it is a single test year with no correction for multiple comparisons.

Repaired minus original graph, same model:

| family | features | r2024doc ΔAP [95% CI] | rall_api ΔAP [95% CI] |
|---|---|---|---|
| logistic | +degree | +.0002 [−.0005, .0010] | −.0001 [−.0010, .0008] |
| logistic | +degree+neighbors | −.0024 [−.0057, .0004] | −.0029 [−.0060, .0001] |
| HGB | +degree | −.0006 [−.0018, .0006] | +.0003 [−.0020, .0022] |
| HGB | +degree+neighbors | −.0018 [−.0053, .0009] | −.0026 [−.0073, .0017] |

For ROC-AUC, only logistic +neighbors under rall_api has an interval that excludes 0: −.0015 [−.0028, −.0003]. Full tables are in `out/bootstrap-*.csv`.

## 5. Connected vs isolated (test)
Repair moves about 113 test firms from isolated to connected, so the groups change composition. Shifts in group metrics therefore mostly reflect who is in each group, not model quality.

For the financial-only logistic model, AP is:
- connected: .8143 → .8038 / .8046
- isolated: .8249 → .8367 / .8360

For logistic with neighbors:
- connected: .8144 → .7987 / .7984
- isolated: .8247 → .8363 / .8355

HGB shows the same pattern (neighbors, connected: .8119 → .7950 / .7940). Full table: `out/group-metrics-all.csv`.

## 6. Interpretation (exploratory)
- Repairing the edges densifies the 2024 graph a lot: about 10% fewer isolated test firms.
- It does not make the network features useful. Degree adds nothing in any variant, and the neighbor-mean block is neutral to slightly negative.
- The neighbor block gets slightly worse after repair, and the result is the same whether the repair is test-year-only (r2024doc) or consistent across years (rall_api). So the train/test graph-construction mismatch in r2024doc is not what drives the small drop.
- A likely contributor is that more connected firms now have no neighbor financials (53 → about 115), so more rows rely on imputed neighbor means.
- The baseline conclusion stands: in this cohort and this test year, the shareholding-network features show no association with better prediction beyond the financial features.
