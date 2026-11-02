# Phase 2B step: restoring the as-filed (as-of) financial samples

**Scope.** This is exploratory and shows association only. The FY2024→FY2025 test year had already been viewed in earlier steps, so it is not an untouched holdout.

**Code.** Everything is in `reruns/asfiled-financials/` (`src/build_panel_asfiled.py`, `src/train_eval.py`, `src/analyze.py`). The baseline and the earlier reruns were not modified. No API calls were made, no GNN was used, and the only new feature is the one indicator in §4(b).

## 1. The single change
The graph is r2024doc (the 361 verified edges, 2024 only). Labels, hyperparameters (logistic C=1, the fixed small HGB, no search), preprocessing and the 2,193 test firms are all unchanged. What changes is that the samples the baseline dropped for late label versions are put back, using as-filed values from `data/processed/asof_financials.csv`.

**Label for a restored sample.** The label is the next-FY OFS operating income < 0, taken from the latest version filed on or before the label cutoff:
- training: 20240430;
- validation: 20250430.

These are the same cutoffs the protocol already imposes on label availability. The source is the document value where available, otherwise XBRL. A sample with no version filed by the cutoff, or with no operating income in that version, stays excluded.

**Features of restored samples.** Their feature-year values are already as-filed by C under the baseline rule (feature disclosure ≤ C). They are unchanged.

| | baseline / r2024doc | restored | not restorable | label flips vs final version |
|---|---|---|---|---|
| training (feature FY2021 / FY2022) | 3,607 | +14 / +62 → **3,683** | 1 (no as-of OI) | **1 of 76** (00308896, FY2022→23: final 0, as-filed 1) |
| validation (feature FY2023) | 2,071 | +303 → **2,374** | 11 of 314 | **0 of 303** |
| test (feature FY2024) | 2,193 | unchanged | | |

Because the training set grew, all models were refit with the same fixed hyperparameters, and the thresholds were re-chosen on the 2,374 validation rows by the same rule (validation max-F1, largest tie). Test thresholds:
- logistic: financial .3308→.3504, +degree .3537→.3481, +neighbors .3824→.3451;
- HGB: financial .3102→.3808, +degree .3521→.4009, +neighbors .4202→.2801.

## 2. Paired comparison with r2024doc (same 2,193 test firms; 500 resamples, seed 20261102)
| family | features | AP r2024doc → as-filed (Δ [95% CI]) | ROC-AUC (Δ [95% CI]) | Brier |
|---|---|---|---|---|
| logistic | financial | .8215 → .8205 (−.0010 [−.0018, −.0002]) | .8868 → .8866 (−.0001 [−.0006, .0003]) | .1247 → .1249 |
| logistic | +degree | .8217 → .8209 (−.0008 [−.0020, .0003]) | .8866 → .8867 | .1249 → .1249 |
| logistic | +degree+neighbors | .8196 → .8207 (+.0011 [−.0004, .0027]) | .8847 → .8856 | .1257 → .1252 |
| HGB | financial | .8360 → .8340 (−.0020 [−.0069, .0024]) | .8939 → .8916 (−.0022 [−.0045, −.0002]) | .1197 → .1204 |
| HGB | +degree | .8355 → .8345 (−.0010 [−.0049, .0029]) | .8933 → .8915 | .1201 → .1207 |
| HGB | +degree+neighbors | .8297 → .8263 (−.0034 [−.0080, .0013]) | .8911 → .8881 (−.0030 [−.0059, −.0001]) | .1207 → .1220 |

## 3. Connected test firms with no usable neighbor financials
- **Main comparison: 114 → 114, unchanged.** This step does not change the neighbor-feature source for the test set: all 0 test neighbor-feature cells differ from r2024doc. The restored samples are validation and training firms. The test neighbors' FY2024 features come from the feature-year panel, which this step does not touch.
- **Labelled second variant, `nbasof` (used only for this count and for sensitivity).** Firms whose feature-year financials were corrected late are allowed to serve as neighbors, using their as-of values dated ≤ C. This is only possible for FY2023 (C=20240430), which adds 63 neighbor nodes, and FY2024 (C=20250430), which adds 273. They are never added as samples.
  - Effect: **114 → 41** firms with no usable neighbor financials; firms with any neighbor missing go 388 → 164; 277 test rows change neighbor features.
  - Test AP with neighbors under nbasof: logistic .8187, HGB .8255. Both are slightly below the main variant.

## 4. Sub-analyses (main variant)
**(a) Only test firms whose neighbors all have usable financials (n=849, 265 positive).** Neighbors minus financial:
- logistic: AP .8206 vs .8160, Δ +.0045 [−.0075, .0157]; ROC .8867 vs .8867.
- HGB: AP .8104 vs .8186, Δ −.0082 [−.0234, .0058]; ROC .8877 vs .8926.
- Under nbasof (n=1,073): logistic −.0011 [−.0124, .0110]; HGB −.0068 [−.0194, .0052].

**(b) Missing-neighbor indicator.**
- The baseline has no explicit indicator. It has `neighbor_financial_fraction`, and the imputer's `add_indicator=True` implicitly flags rows whose neighbor means are all missing, which covers isolated firms and firms with no usable neighbor.
- I added one explicit indicator, `any_neighbor_missing_financials` (degree>0 and usable neighbors < degree), as the labelled condition `financial_degree_neighbors_missind`.
- Logistic: AP .8208, which is +.0001 [−.0009, .0010] vs neighbors and +.0003 [−.0048, .0048] vs financial.
- HGB: its predictions are identical to the neighbors model. The fixed small trees never split on the indicator, so Δ = 0.

## 5. The 113 newly connected firms (isolated in the original graph, connected after repair; 42 positive)
FP / FN at each model's own threshold, for the graph-only repair (r2024doc) and after also restoring the as-filed samples:

| model | original graph | r2024doc | as-filed (main) |
|---|---|---|---|
| logistic financial | 14 / 12 | 14 / 12 | 13 / 12 |
| logistic +neighbors | 13 / 12 | 12 / 13 | 15 / 11 |
| HGB financial | 17 / 10 | 17 / 10 | 14 / 11 |
| HGB +neighbors | 13 / 15 | 13 / 15 | 19 / 6 |

Rates use 71 negatives and 42 positives. For example, HGB +neighbors goes from FP 18% / FN 36% to FP 27% / FN 14%. Most of that shift comes from its threshold falling from .4202 to .2801, not from a change in ranking.

By the repaired grouping (as-filed, HGB +neighbors; r2024doc → as-filed):

| group | n | positive | FP | FN |
|---|---|---|---|---|
| isolated | 956 | 394 | 92 → 163 | 91 → 48 |
| connected, all neighbors have financials | 849 | 265 | 59 → 112 | 66 → 43 |
| connected, some neighbor missing financials | 388 | 97 | 27 → 46 | 30 → 16 |
| newly connected with all neighbor financials | 41 | | 3 → 6 | 6 → 3 |
| newly connected with missing neighbor financials | 72 | | 10 → 13 | 9 → 3 |

Logistic +neighbors moves much less, for example connected-all 59/72 → 71/59. The full table for all models and variants is in `out/fp-fn.csv`.

## 6. Decision rule (fixed before seeing the results), applied mechanically
The test is whether the AP difference (neighbors minus financial) has a 95% paired-bootstrap interval containing 0. This uses the main variant and the baseline's bootstrap stream (`out/main/paired-bootstrap.csv`):

| family | ΔAP [95% CI] | contains 0? |
|---|---|---|
| logistic | +.0002 [−.0050, .0047] | yes |
| HGB | **−.0078 [−.0146, −.0009]** | **no, it excludes 0 in the negative direction** (neighbor model worse) |

Recomputed with an independent resampling stream, the HGB interval is [−.0152, −.0012] and still excludes 0.

**The condition "both intervals contain 0" is not met.** So, as the rule requires, the line is not declared closed and `NODE_PREDICTION_SUMMARY.md` was not written. The only finding stated is this one: HGB with neighbor features has lower test AP than HGB with financial features only, in this one test year. Nothing beyond that is concluded.

## 7. Limitations
- Survivor/observed-cohort bias: only firms with KRW, December-year-end financials and an observed disclosure are included.
- A single test year, already viewed, and multiple comparisons with no correction.
- The 361 verified edges cover 2024 only, so the training graphs and the test graph are built differently.
- Neighbor financials are often unavailable (114 connected test firms with no usable neighbors; 41 even under nbasof).
- The as-of label is document or XBRL based. 11 validation and 1 training sample could not be restored.
- Association only; nothing here is causal.
