# 确认性测试协议（FY2025→FY2026，2027 年执行）

Frozen on 2026-11-03 KST. After the commit, changes go only in an appended amendments section, with the date, time and reason. **Pending Jack** marks an open decision.

**Status:** all earlier test results, including 2024→2025, were viewed and are exploratory. The only confirmatory test is FY2025→FY2026, evaluated **once**.

## 数据
- **Data that do not exist yet** (available in 2027):
  - the FY2026 사업보고서 (C_2026 = 2027-04-30);
  - the FY2026 investment rows, financials and filing list.
- **Collection.** These are collected after 2027-04-30 with the same API calls (calls are counted in `data/state/calls.json`). The rest is unchanged:
  - eligibility: an annual report filed on or before C;
  - holding conditions;
  - unique-match resolution (exact / norm / stock name);
  - former names (rule B, as filed on or before C);
  - the A2 trailing-token rule, with exactly the token list in A2.
- **Feature graph G_t.** Rows filed on or before C_t only.
- **History H_t (label side only).** Includes corrections and rule A.
- **Document recovery: pending Jack.** It must apply the same way to every year. The options are in LINK_STAGE1B §2.
  - Default recommendation: (a) API rules for every year, with no document recovery.
  - If Jack chooses document recovery, it must also be done for every year 2019–2026, which needs on the order of 10k calls.
- **Industry and market.** Current values (as-of values are not available, A3), fixed from a snapshot taken at the time of the confirmatory run. The sensitivity analysis (M1 without industry or market) is reported too.

## 划分
- **Train:** 2021→22, 2022→23, 2023→24.
- **Validation (model selection only):** 2024→25.
- **Confirmatory test:** 2025→26.

## 负样本 / 模型 / 网格 / 种子
- **Negatives:** as in protocol.md §4, investor-anchored, 100 per positive, seed 20261103.
- **Models and grids:** as in protocol.md §7.
  - M1 and M1+scores: C ∈ {0.01, 0.1, 1, 10}, fitted on the train years only.
  - Heuristics: CN, AA, Jaccard and PA; the best one on validation enters S1 and S2.
  - M3: grid as in §7, seeds {1,2,3,4,5}.
- **Bootstrap:** 1000 resamples clustered by investor, seed 20261103, 95% percentile interval.

## 子集 / 分组
- **Subsets:** all new edges; non-financial investor (KSIC 64–66 count as financial) with ownership ≥ 5%.
- **Groups (degree in G_t):** both connected / one isolated / both isolated.
- A cell with fewer than 30 test positives is reported as not informative.

## 成功标准（照抄）
S1 图方法相对 M1 的 AP 差，95% 区间不含 0；S2 GNN 相对最强启发式的 AP 差，95% 区间不含 0。

The S1 comparisons are (i) the best heuristic − M1 and (ii) M1+scores − M1, with the result for all new edges as the main result.

## 主指标
- **Test AP stays the primary metric**; Hits@10, Hits@50 and MRR are secondary.
- **Reasons:**
  1. AP was pre-registered.
  2. It uses all sampled pairs, so it is more stable than Hits@K. Only about 140 investors have positives, and Hits@K jumps in steps of 1/140.
  3. Hits@K depends on how the full ranking is filtered and on ties.
- **Disclosure:** this recommendation was written **after viewing** the 2024→2025 results (both the interim and the Stage 1B ones). In those results M1+scores was stronger on Hits@50 than on AP. Keeping AP means the primary metric was not changed after the data were seen.

## 测试评估规则
- The test is run once, by a single command.
- The script refuses to run if the test output file already exists.
- Nothing is changed after the results come out. Any problem found later is recorded in an amendment and treated as exploratory.
