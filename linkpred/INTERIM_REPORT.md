# 链路预测中期报告（M0–M2）

The protocol is in `protocol.md` (commit 285f96a) and amendment A1 was added in 108f1b6. Validation choices were frozen and committed before the test run. The test set was evaluated **once**, at 2026-11-03 KST; the results are frozen in `out/test_frozen.json`. **M3 has not been started; waiting for Jack's confirmation.**

## 数据
| FY | C | 合格节点 | 严格 as-of 有向边 | 文档核实补边 |
|---|---|---|---|---|
| 2019 | 20200430 | 2415 | 233 (名称解析问题，未验证) | 0 |
| 2020 | 20210430 | 2500 | 265 (同上，未验证) | 0 |
| 2021 | 20220430 | 2606 | 1192 | 0 |
| 2022 | 20230430 | 2701 | 1480 | 0 |
| 2023 | 20240430 | 2796 | 1663 | 0 |
| 2024 | 20250430 | 2799 | 1932 | 282 (361 对中满足合格/去重后新增) |
| 2025 | 20260430 | 2793 | 1919 | 0（2025 未做文档核实，未验证） |

- **Positives** (first appearance, undirected, both endpoints eligible in t and t+1): 2021→22: 289; 2022→23: 292; 2023→24 (validation): 358; 2024→25 (test): 333. Each has 100 negatives; no positive had fewer than 100 candidates.
- **Non-financial investor and ≥5% subset:** 54 / 62 / 120 / 45.
- **Test groups:**
  | Test cell | Both connected | One isolated | Both isolated |
  |---|---|---|---|
  | All new edges | 243 | 80 | 10 (<30, too few to be informative) |
  | Subset (non-financial, ≥5%) | 20 | 20 | 5 (all <30, not informative) |
- **Unit checks** (`out/unit_checks.csv`): all pass.
  - Financial disclosure dates are ≤ C.
  - No positive appears in the history H_t.
  - Negatives are not in H_t or G_{t+1}.
  - Every positive is in G_{t+1}.
  - Every positive has exactly 100 negatives.

## 验证集选择（只用验证集）
- M1: C=0.01, validation AP .0326.
- M1+scores: C=0.01, validation AP .0637.
- The best single heuristic is **Jaccard** (validation AP .0471). The others: AA .0296, CN .0203, PA .0180.
- M0 validation AP: random .0104, same industry .0166, same market .0132.
- Note: C sits at the lower edge of the grid. The grid was not widened, as the protocol requires.

## 测试（仅一次）AP，全部边（333 正 / 33,633 对）
| Model | AP | Diff vs M1 [95% firm-clustered bootstrap CI] |
|---|---|---|
| Random | .0114 | |
| Same industry | .0126 | |
| Same market | .0105 | |
| CN | .0219 | |
| AA | .0285 | |
| **Jaccard (best on validation)** | .0207 | −.0255 [−.0613, +.0202] |
| PA | .0275 | |
| **M1** | .0462 | — |
| **M1+scores** | .0617 | +.0154 [−.0019, +.0487] |

**S1 verdict: not met.** Neither pre-registered graph method has a 95% interval vs M1 that excludes 0 on all new edges.

**Subset: non-financial investor and ≥5% (45 positives, informative):**
- M1 .0176; M1+scores .1102; Jaccard .1007; AA .1199.
- M1+scores − M1: +.0927 [+.00002, +.2075]. The interval only just excludes 0, and this subset is not the primary analysis.
- Jaccard − M1: +.0831 [−.0094, +.1929].

**Groups (all edges):**
- **Both connected (243):** M1 .0625, M1+scores .0788 (Δ +.0163 [−.0096, +.0690]), Jaccard .0286.
- **One isolated (80):** all heuristics are 0, so they cannot rank these pairs. M1 .0119; M1+scores −.0001 [−.0041, +.0032]; Jaccard − M1 −.0067 [−.0231, −.0012].
- **Both isolated:** fewer than 30 positives, so not informative.

## 测试全排序（每投资方，133 位投资方，346,356 候选）
| Model | Hits@10 | Hits@50 | MRR |
|---|---|---|---|
| Random | .008 | .060 | .005 |
| Same industry | .015 | .060 | .007 |
| CN | .090 | .120 | .046 |
| AA | .105 | .120 | .045 |
| Jaccard | .090 | .113 | .041 |
| PA | .060 | .105 | .053 |
| M1 | .030 | .150 | .024 |
| M1+scores | .113 | .173 | .045 |

## 局限 / 未验证
1. **FY2019–2020 edges:** about half of the investee names carry a (상장)/(비상장)-type prefix that the resolver misses, so these graphs are very sparse. This weakens the "never appeared before" check, and early "first appearances" may be overstated (未验证).
2. **Non-2024 years used API rules only:** they were not document-verified (about 8k calls would be needed, 未验证). The 2024 graph is therefore denser than the other years, which affects the validation label and the test history in a non-uniform way.
3. **Industry and market:** these are current values, not as-of values (possible leak, 未验证). The financial-investor definition (KSIC 64–66) is an executor choice.
4. **Survivor bias:** only firms filing an annual report in both years are included. For firms listed later, a "first appearance" may only mean first observed.
5. **Correction filings:** features exclude corrections filed after C. They are used only on the history (label) side.
6. **Absolute AP is very low** (base rate about 1%). Only 333 test positives spread over 133 investors give wide intervals.

Reproduce: `../.venv/bin/python src/build_data.py`; `../reruns/repaired-edges/.venv39/bin/python src/models.py` (validation). `--test` has already been run once, and the script refuses to run it again.

## 更正（2026-11-03 KST，Stage 1B）
- The yearly edge counts in the 数据 table above (1,192 … 1,919) are **total** strict as-of directed edges per year, not first-appearance edges. My earlier summary called them "edges per year", which was ambiguous.
- The first-appearance undirected edges and the sample flow are in `../reports/LINK_STAGE1B.md`.
- The results in this report predate the A2 and A3 fixes. See Stage 1B for the repaired numbers.
