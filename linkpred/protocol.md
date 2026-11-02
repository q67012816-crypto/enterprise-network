# Pre-registered protocol: first-appearance link prediction on the listed-company shareholding graph

Registered on 2026-11-03 (KST). Committed before any dataset statistics were built and before any model was run. **This text must not be edited after the commit.** Any change goes in an appended section, `## Amendments`, with the date and time (KST) and the reason. Choices not fixed by Jack are marked **[executor choice]**. Anything not verified is marked **未验证**.

## 1. Data
- **Nodes.** The KOSPI+KOSDAQ universe in `data/processed/universe_firms.csv`. A firm is *eligible in FY t* if its FY t annual report (사업보고서) was filed on or before C_t = (t+1)-04-30. **[executor choice]** A pair is a candidate for t→t+1 only if both firms are eligible in FY t and in FY t+1.
- **Edge source.** OpenDART "investments in other corporations" rows for the 사업보고서 of FY 2019–2025 (`data/interim/invest_panel_raw.csv`).
- **Investee name → corp_code resolution.** Uses the existing rules: exact name, then normalised name, then stock name, with a unique match required. Former names are added through as-filed name history only when the match is unique and the name was filed on or before C_t (the "rule B" used in Phase 2B).
- **A row counts as an edge** only if all of the following hold:
  - API status is 000;
  - the period-end holding is > 0 (percentage in (0,100], or quantity > 0, or book value > 0);
  - the percentage is valid;
  - investor ≠ investee;
  - both are eligible nodes.
- **Feature graph G_t (strict as-of).** Only rows from filings filed on or before C_t.
  - **FY2024 only:** add the 361 edges verified against the original filing (`data/processed/doc_recovered_edges_2024.csv`).
  - **Other years:** the API-rule edges only. Verifying the original documents for 2019–2023 and 2025 would need about 8k API calls and was not done (**未验证**).
  - **Consequence:** the 2024 graph is built more completely than the other years. API rows that exist only in a correction filed after C ("rule A") are **not** used for features in any year, because their content dates after C.
- **History set H_t (label side only) [executor choice].** The union of all edges for FY 2019..t under any rule, including rule A and corrections filed up to the data snapshot. It is used only to decide whether an edge is "new"; it is never used as a feature. This is conservative: a pair that was missed earlier only because a correction was disclosed late is not counted as a new edge.
- **Financial features (FY t, disclosure ≤ C_t).**
  - The OFS panel rebuilt from `fin_multi_raw` (`reruns/repaired-edges/fin_panel_rebuilt.csv`), KRW, with assets > 0. The latest financial disclosure for FY t must be ≤ C_t; otherwise the firm's financials are missing.
  - Features: log assets; liabilities, equity, capital, revenue, operating income and net income each divided by assets; signed log revenue; signed log operating income; and the current operating-loss flag.
  - Missing values: median imputation (training medians) plus a missing indicator; clipping at the training 0.5/99.5 percentiles.
- **Industry.** The first 2 digits of `induty_code` (KSIC division). **Market:** `corp_cls` (Y=KOSPI, K=KOSDAQ). Both come from the *current* company overview, not as-of values (**未验证**, a possible leak: a firm may have changed industry or market).
- **Financial investor [executor choice].** KSIC divisions 64, 65 or 66.

## 2. Task and label
- Given G_t and the FY t financials, predict the undirected pairs {u,v} that are edges in G_{t+1} (strict as-of C_{t+1}) and are **not** in H_t, i.e. they appear for the first time in FY t+1.
- Edge direction is merged when deduplicating undirected pairs and is kept as an attribute (investor = source).
- For each positive, the **anchor/investor** is the source of the new directed edge. If both directions are new, the anchor is the one with the smaller corp_code **[executor choice]**.
- The ownership percentage of a positive is the maximum over its new directed rows.

## 3. Splits
- **Train:** 2021→2022 and 2022→2023.
- **Validation:** 2023→2024. All model and hyperparameter selection uses validation only.
- **Test:** 2024→2025, evaluated **once**.
  - The M0–M2 test numbers are computed once at the end of the interim step and then frozen.
  - M3's test evaluation also happens once, later.
  - S2 uses the frozen M2 numbers.
- 2019–2020 are used only for H_t.

## 4. Negatives
- **100 negatives per positive**, seed 20261103.
- **Sampling rule [executor choice], investor-anchored:** keep the positive's anchor u and draw v uniformly, without replacement, from eligible nodes v ≠ u such that {u,v} is not in H_t ∪ G_{t+1}. Negatives are therefore never past edges or that-year edges.
- Both endpoints of every pair satisfy the eligibility rule in §1.
- If fewer than 100 valid v exist, use all of them and report it.
- The same sampled pairs are used for every model.
- **Ranking evaluation.** For each investor with ≥1 positive in t+1, rank **all** eligible candidates v (excluding H_t ∪ existing pairs, but not excluding that investor's positives). Report Hits@10, Hits@50 and MRR. MRR is the mean of 1/rank of the best-ranked positive per investor **[executor choice]**. Also report the mean over positives of the reciprocal rank, where a positive is ranked against non-positive candidates only.

## 5. Metrics and inference
- **Primary metric:** test AP on the sampled pairs. Secondary: ROC-AUC, plus the ranking metrics above.
- **Paired cluster bootstrap:** 1000 resamples, rng `numpy.default_rng(20261103)`. Anchor/investor firms are resampled with replacement, and all of each firm's pairs (its positives and their negatives) go along with it. The interval is the 95% percentile interval of the AP difference.
- **Success criteria (verbatim):**
  - S1: AP difference of graph methods relative to M1 has a 95% interval that excludes 0;
  - S2: AP difference of the GNN relative to the strongest heuristic has a 95% interval that excludes 0.
- **For S1 [executor choice],** the "graph methods" are exactly two pre-specified comparisons, each reported separately with no multiplicity correction:
  - (i) the best single M2 heuristic, chosen by validation AP;
  - (ii) M1 + heuristic scores.
- **For S2,** "strongest heuristic" means the single heuristic with the best validation AP, compared on test using its frozen scores.

## 6. Subsets and groups
- **Subsets:**
  - all new edges;
  - new edges whose investor is non-financial **and** whose ownership is ≥ 5%. A subset positive keeps its own 100 negatives.
- **Groups**, by the endpoints' degree in G_t: both endpoints connected / one endpoint isolated / both isolated.
- **Minimum size:** a cell with fewer than 30 test positives is reported as "too few to be informative". The threshold is not relaxed.

## 7. Models (same sampled pairs)
- **M0:**
  - random scores (seed 20261103);
  - same industry (1/0);
  - same market (1/0).
- **M1:** logistic regression on [anchor features, other-endpoint features, same-industry, same-market, the two industries one-hot (top 20 divisions + other), the two markets]. No graph features. Standardised; lbfgs; `class_weight=None`. C is chosen on validation AP from {0.01, 0.1, 1, 10}.
- **M2:** common neighbours, Adamic–Adar, Jaccard and preferential attachment, computed on the undirected G_t.
  - As standalone scores, the best one is chosen on validation.
  - M1 + scores: the M1 design plus log1p(CN), AA, Jaccard and log1p(PA), with C chosen on validation from the same grid.
- **M3 (not run until Jack confirms):** a GraphSAGE link predictor on G_t (undirected, mean aggregator), with financial and industry/market node features prepared the same way as M1 and a dot-product or MLP decoder.
  - Grid: hidden {64,128} × layers {1,2} × lr {1e-3,1e-2} × dropout {0,0.3} × decoder {dot, MLP}.
  - Adam, weight decay 0, up to 200 epochs, early stopping on validation AP with patience 20.
  - Training uses the train-split positives and sampled negatives. Message passing for year t uses G_t only.
  - Seeds {1,2,3,4,5}. The configuration is chosen by mean validation AP across the 5 seeds. The test result is the mean of the 5 seed predictions' AP; the bootstrap uses the seed-averaged score.

## 8. Order
1. Build the data and report counts (positives per year, subset and group) before any modelling.
2. Develop M0–M2 on validation.
3. Freeze the choices.
4. Evaluate test once.
5. Write `INTERIM_REPORT.md`.
6. **Stop and wait for Jack's confirmation before M3.**

## 9. Leakage checks and risks
- **Unit checks:**
  - every feature filing date is ≤ C_t;
  - no positive is in H_t;
  - negatives are not in H_t ∪ G_{t+1};
  - validation and test choices do not touch test labels.
- **Known risks:**
  - Edges are disclosed after C, and corrections change apparent appearance timing. The H_t rule mitigates this, but it remains **未验证** for years other than 2024.
  - Survivor bias: only firms filing an annual report in both t and t+1 are included.
  - A firm listed later has no earlier history, so its "first appearance" may only mean first observed.
  - Industry and market are current values.
  - Name resolution can fail, which produces missing edges.
  - The 2024 doc-verified edges make the 2024 graph and labels denser than the other years. In particular, validation 2023→24 positives include verified edges that the API-only 2023 graph may have missed; H_2023, which includes rule A, partly mitigates this.
  - Results are association only and say nothing causal.

## Amendments
- **2026-11-03 (KST), A1.** This clarification fills a gap the protocol left open. It was written after the dataset counts were built and **before any model was fitted or scored**.
  - **Model fitting.** M1 and M1+scores are fitted on the train pairs (2021→22 and 2022→23) only. The model chosen on validation is applied to test as is, with no refit on train+validation.
  - **Ties in the ranking metrics.** These use the expected rank under random tie-breaking: rank = #higher + (#tied + 1)/2. A hit at k means this rank is ≤ k.
  - **Group AP.** Computed on all pairs whose own group label (negatives included) is in that group.
  - **Data limitation found while building.** In FY2019–2020, about half of the investee names carry a (상장)/(비상장)-type prefix that the existing resolver does not strip. As a result, G_2019 and G_2020 have only 233 and 265 edges, against 1,192 for 2021. This weakens H_t (the "never appeared before" check) for 2019–2020 history. The resolver was left unchanged, as the protocol requires, and the issue is recorded as **未验证**.
- **2026-11-03 (KST), A2: investee-name prefix/suffix normalisation.**
  - **Reason.** In FY2019–2020, about 47% of the investee names (`inv_prm`) end in a listing-status tag, e.g. `(비상장)` (7,932 rows in 2019) or `(상장)` (1,063), so they could not be resolved. Many later-year names carry the same tags or footnote markers.
  - **Rule (`src/build_v2.py::a2`).** Strip trailing tokens repeatedly, but only when a token *exactly* matches one of the following, then pass the name to the unchanged exact / norm / stock-name resolver. A unique match is still required, an ambiguous match is still rejected, and there is no fuzzy matching.
    - In round brackets: `(비상장)`, `(상장)`, `(시장성)`, `(비시장성)`, `(해외비상장)`, `(해외 비상장)`, `(해외상장)`, `(해외 상장)`, `(*)`, `(**)`, `(*N)`, `(주N)`.
    - In square brackets: `[비상장]`, `[상장]`, `[해외 비상장]`, `[해외 상장]`.
    - N stands for digits. `(주)` (주식회사) is still handled by the existing normaliser.
  - **Strict as-of directed edges per year, before → after:** 2019 233→835; 2020 265→944; 2021 1192→1324; 2022 1480→1628; 2023 1663→1848; 2024 1932→2110 (including the doc-verified additions; 1650→1828 without them); 2025 1919→2105.
  - **Affected:** every graph G_t, every history set H_t, all positives and negatives, and every M0–M2 result. The old files are kept in `data/`, with results in `out/`. The new files are in `data_a2/`, with results in `out_1b/`.
  - **Interpretation.** The 2024→2025 test was already viewed before this fix, so every test number after A2 is **exploratory**.
- **2026-11-03 (KST), A3: as-of industry and market.**
  - **Finding.** Historical values cannot be obtained from the cached data or the DART API.
    - `company.json` returns only current values.
    - The `corp_cls` field in `filings_A001.csv` (2,611 firms) and in `invest_panel_raw.csv` never changes within a firm, which shows it is a current snapshot.
    - There is no cached source for a historical `induty_code`.
    - The original documents cover mainly FY2024 (3,193 of 3,376), which is not enough.
    - No API calls were made.
  - **Rule.** The main analysis keeps the current values, labelled "current value (not as-of, 未验证)". As a sensitivity analysis, M1 and M1+scores are run with industry and market removed (`out_1b_noindmkt/`). M0 same-industry and same-market necessarily still use the current values.
- **2026-11-03 (KST), A4: graph construction made consistent across years (option (a)).**
  - **Reason.** Link prediction needs every year's graph to be built the same way. Only 2024 had document recovery, which made 2024 denser than the other years, and 65 of the 368 validation positives were edges that existed only in the document-recovered set.
  - **Rule.** Every year (2019–2025) uses API rules only, with no document recovery. The A2 name rule and the A3 industry/market handling (current values) are kept. Data: `data_a2_apionly/`. Results: `out_1c_a4/`.
  - **Interpretation.** A4 was decided **after viewing** the earlier 2024→2025 test results (the interim run and Stage 1B), so all test results after A4 are **exploratory**.
  - The protocol's model-selection rule (validation AP) and grids are unchanged.
- **2026-11-03 (KST), A5: post-hoc within-candidate-set metrics (reported only; does not change this protocol's primary metric).**
  - **Metrics.** Each positive is ranked among its own 101 candidates (1 positive + its 100 negatives), with ties by expected rank as in A1. Reported: within-set AUC, MRR and Hits@10, each with a 1000-resample paired bootstrap clustered by investor.
  - **Post-hoc.** These metrics were added **after viewing** the 2024→2025 test results.
- **Naming note.** This original text uses "Jack" in several places, and it is not edited. "Jack" and Mingsen Zhang are the same person. The confirmatory protocol `protocol_confirm.md` uses Mingsen Zhang.
- **Column note.** The `label_doc_only` column produced by `build_v2.py` has been renamed `also_in_2024_doc_set`. It only marks that a pair is also in the 2024 document-recovered set; it **does not** by itself mean the label is contaminated.
