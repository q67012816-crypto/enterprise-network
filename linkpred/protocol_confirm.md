# 确认性测试协议（FY2025→FY2026，2027 年执行）

Frozen on 2026-11-03 KST. After the commit, changes go only in an appended amendments section, with the date, time and reason. Mingsen Zhang makes the decisions on this protocol.

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
- **Document recovery: decided, option (a)** (protocol.md A4, chosen by Mingsen Zhang on 2026-11-03 KST). Every year uses API rules only, with no document recovery. **This was decided after viewing the 2024→2025 results.**
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
Original text (protocol.md): S1 图方法相对 M1 的 AP 差，95% 区间不含 0；S2 GNN 相对最强启发式的 AP 差，95% 区间不含 0。

**Confirmatory version (changed after viewing; the metric was switched to within-group AUC):**
- S1: the within-group AUC difference of graph methods relative to M1 has a 95% interval that excludes 0.
- S2: the within-group AUC difference of the GNN relative to the strongest heuristic has a 95% interval that excludes 0.
- **Comparisons:** S1 compares (i) the best heuristic − M1 and (ii) M1+scores − M1, with the result for all new edges as the main result. "Best heuristic" means the best on validation, chosen by within-group AUC.
- **Interval:** paired bootstrap, 1000 resamples clustered by investor, seed 20261103, 95% percentile interval.

## 主指标（看过结果后修改）
- **Primary: within-group AUC.** Each positive is ranked among its own 101 candidates (1 positive + its 100 negatives), with ties by expected rank. The metric is the mean over positives.
- **Secondary:** pooled AP, within-group MRR and Hits@10, and full-ranking Hits@10/50 and MRR.
- **Model selection** (C, best heuristic, and M3 hyperparameters) also uses validation within-group AUC.
- **Reason:** negatives are anchored per investor. Pooled AP mixes pairs from different investors in one ranking, so it is affected by score-level differences between investors, such as an investor-level effect. Within-group AUC compares only candidates of the same investor, which matches the task of "which firm will this investor connect to".
- **Disclosure:** this decision was made by Mingsen Zhang **after viewing** the 2024→2025 results (interim, Stage 1B and Stage 1C), and is a post-hoc change. An earlier version of this file kept AP as the primary metric.

## 测试评估规则
- The test is run once, by a single command.
- The script refuses to run if the test output file already exists.
- Nothing is changed after the results come out. Any problem found later is recorded in an amendment and treated as exploratory.

## 修订记录
- **2026-11-03 (KST), per Mingsen Zhang.** All changes below were made after viewing the 2024→2025 results.
  1. The primary metric changed from AP to within-group AUC, and S1/S2 were reworded to match.
  2. The document-recovery policy is fixed as option (a).
  3. "Jack" was replaced with Mingsen Zhang. The original `protocol.md` is not edited; there, the naming is explained in an amendment.
