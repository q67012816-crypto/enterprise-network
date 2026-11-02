# 链路预测 Stage 1B 报告（2026-11-03 KST）

**M3 has not been started.** **The 2024→2025 test was already viewed in the interim step, before these repairs, so every test number in this report is exploratory.** They are not a confirmatory test of S1 or S2.

Git author: the local config for this repo only is now `Mingsen Zhang`. There was no local email before, so I used `mingsen-zhang@users.noreply.invalid`, a placeholder address that does not exist. Old commits were not rewritten.

## 1. 边数澄清与样本流转
In the interim report, 1,192 … 1,919 are the **total** strict as-of directed edges for each year, not the number of new edges. A correction has been appended to INTERIM_REPORT.md.

**Undirected edges appearing for the first time, by FY** (absent from every earlier year's history 2019..t−1; history includes corrections and rule A):

| FY | Total undirected (old) | First appearance (old) | Total undirected (A2) | First appearance (A2) |
|---|---|---|---|---|
| 2020 | 262 | 108 | 923 | 280 |
| 2021 | 1173 | 951 | 1293 | 562 |
| 2022 | 1439 | 337 | 1576 | 307 |
| 2023 | 1618 | 324 | 1792 | 326 |
| 2024 | 1865 | 423 | 2026 | 435 |
| 2025 | 1847 | 386 | 2018 | 394 |

- 2019 is the first year, so first appearance is not defined for it.
- The old 2021 figure of 951 was inflated by the 2019–2020 name problem.

**Sample flow, old data → positives (289 / 292 / 358 / 333):**

| Step | 21→22 | 22→23 | 23→24 | 24→25 |
|---|---|---|---|---|
| FY t+1 resolved held rows (any disclosure date) | 1835 | 1916 | 2107 | 2250 |
| Disclosed on or before C_{t+1} | 1666 | 1795 | 1787 | 2019 |
| Both endpoints eligible in FY t+1 | 1485 | 1667 | 1657 | 1927 |
| Directed dedup (+ doc additions for 2024) | 1480 | 1663 | 1932 | 1919 |
| Undirected dedup (direction merged) | 1439 | 1618 | 1865 | 1847 |
| − Already in history (not first appearance) | −1102 | −1294 | −1442 | −1461 |
| = First appearance | 337 | 324 | 423 | 386 |
| − An endpoint not eligible in FY t | −48 | −32 | −65 | −53 |
| **= Positives** | **289** | **292** | **358** | **333** |
| Memo: both endpoints have financials by C (not a filter; missing values imputed) | 111 | 159 | 293 | 257 |
| Memo: financial investor | 93 | 90 | 71 | 143 |
| Memo: non-financial investor and ≥5% | 54 | 62 | 120 | 45 |

With A2, the same flow gives 260 / 290 / 368 / 339 positives. Full table: `linkpred/data_a2/sample_flow.csv`.

## 2. 验证集污染（只报告，未执行）
**Direct effect on validation:**
- With the old rules, **69 of the 358** validation positives (19%) are edges that exist only through the 2024 document recovery, i.e. they are absent from the API-only 2024 graph.
- With A2 it is 65 of 368 (18%).
- So the validation labels are substantially affected by the extra edges.

**Indirect effects:**
- **Test:** the document recovery adds only 5 pairs to H_2024, because most recovered pairs were already in the history through rule A or other years. **No test positive is removed** (339 either way).
- **Test groups:** the recovered edges do change G_2024 degrees and the heuristic scores. In A2, 11 test positives move into "both connected": with recovery 250 / 79 / 10, without it 239 / 84 / 16.
- **Training:** no effect. Training years do not use the document recovery.

**Options:**

| Option | Validation positives | Test | Train | Pros | Cons |
|---|---|---|---|---|---|
| (a) API rules only for every year, no document recovery | 303 | 339 | 260 / 290 | Every year is consistent and reproducible, with no extra calls | Gives up the 264 verified 2024 edges, so the 2024 graph is incomplete (known misses) |
| (b) Keep the recovered edges in features and history, but drop them from new-edge labels | 303 | 339 (same groups as now) | Same | Features are more complete and the labels match the API rules | Features and labels follow different rules; 2024 features are more complete than other years |
| (c) Document recovery for every year 2019–2026 | 未验证 | 未验证 | 未验证 | Most complete and consistent | About 8–10k calls (half a day's budget or more); timing problems from corrections remain |

Recommendation: (a) or (b) for now, and (c) if budget allows before the confirmatory test. **Pending Jack.**

## 3. 名称前缀修复（Amendment A2）
**Variants found in the data:**
- In 2019, the suffix `(비상장)` appears 7,932 times, `(상장)` 1,063 and `[비상장]` 269. There are also `(비시장성)`, `[해외 비상장]` and footnote markers such as `(*1)` and `(주1)`.
- In 2021, `(비상장)` still appears 1,008 times.

**Rule:**
- Strip trailing tokens repeatedly, but only when they exactly match the listed tokens, then use the unchanged resolver (unique match, ambiguous matches rejected, no fuzzy matching). Full rule in protocol.md A2.
- Applied to every year.
- The old edge files are kept in `linkpred/data/`.

**Directed edges per year, old → new:**

| Year | Old | New |
|---|---|---|
| 2019 | 233 | 835 |
| 2020 | 265 | 944 |
| 2021 | 1192 | 1324 |
| 2022 | 1480 | 1628 |
| 2023 | 1663 | 1848 |
| 2024 | 1932 | 2110 |
| 2025 | 1919 | 2105 |

## 4. 当年行业 / 市场（Amendment A3）
- **Historical values cannot be obtained.** `company.json` gives only current values. `corp_cls` never changes within a firm in either the filing list or the investment rows (2,611 firms), so it is a current snapshot. There is no cached source for a historical industry code. The original documents cover mainly FY2024. No API calls were made.
- **Main analysis:** current values, labelled current value, **未验证**.
- **Sensitivity analysis:** M1 without industry or market.

## 5. 重跑 M0–M2（A2 数据，探索性）
**Sample counts (positives):**

| Split | All | Both connected | One isolated | Both isolated | Non-fin ≥5% (both / one / iso) |
|---|---|---|---|---|---|
| 21→22 train | 260 | 132 | 96 | 32 | 40 (9 / 22 / 9) |
| 22→23 train | 290 | 142 | 113 | 35 | 59 (19 / 30 / 10) |
| 23→24 validation | 368 | 155 | 162 | 51 | 120 (16 / 70 / 34) |
| 24→25 test | 339 | 250 | 79 | 10 | 40 (14 / 22 / 4) |

**Validation choices:**
- M1 C=0.01 and M1+scores C=0.01, the same as before.
- **The best heuristic changed from Jaccard to AA**: validation AP .03624 vs .03619, essentially a tie.

**Test AP, all new edges:**

| Model | Interim (old data) | Stage 1B (A2) | Stage 1B without industry/market |
|---|---|---|---|
| Random | .0114 | .0112 | |
| Same industry | .0126 | .0128 | |
| CN | .0219 | .0132 | |
| AA | .0285 | .0143 | |
| Jaccard | .0207 | .0139 | |
| PA | .0275 | .0281 | |
| M1 | .0462 | .0427 | .0512 |
| M1+scores | .0617 | .0469 | .0504 |
| Best heuristic − M1 [95% CI] | Jaccard −.0255 [−.0613, +.0202] | AA **−.0285 [−.0605, −.0037]** | AA −.0370 [−.1039, +.0015] |
| M1+scores − M1 [95% CI] | +.0154 [−.0019, +.0487] | +.0042 [−.0053, +.0184] | −.0009 [−.0318, +.0229] |

**Subsets and groups (A2, AP difference vs M1):**
- **Non-financial investor, ≥5% (40 positives):**
  - M1 .0227, AA .0099 (Δ −.0128 [−.0480, −.0012]).
  - M1+scores Δ −.0031 [−.0129, +.0057].
  - The interim gain (+.09) disappears: after the repair, many of the old "new edges" turn out to be pairs that already existed in 2019–2020.
- **Both connected (250):**
  - M1 .0575, M1+scores .0590 (Δ +.0015 [−.0133, +.0247]).
  - AA Δ −.0381 [−.0765, −.0026].
- **One isolated (79):** the heuristics are all 0, so they cannot rank these pairs. M1+scores Δ +.0002 [−.0047, +.0023].
- **Both isolated:** 10 positives, not informative.

**Test full ranking (139 investors, Hits@10 / Hits@50 / MRR):**

| Model | Hits@10 | Hits@50 | MRR |
|---|---|---|---|
| Random | .000 | .029 | .003 |
| CN | .072 | .094 | .028 |
| AA | .072 | .094 | .025 |
| Jaccard | .065 | .086 | .026 |
| PA | .058 | .101 | .050 |
| M1 | .036 | .151 | .029 |
| M1+scores | .065 | .209 | .023 |

In the interim results, M1+scores had .098 / .171 / .059.

**S1 status (exploratory):**
- The heuristic did not beat M1. Its interval lies entirely below 0, meaning it is significantly worse than M1.
- M1+scores − M1 includes 0.
- So on the repaired data there is no sign that graph methods beat M1. This does not count as confirming or rejecting S1.
- All leakage unit checks pass.

## 6. 确认测试协议
- `linkpred/protocol_confirm.md` freezes the settings for FY2025→FY2026.
- Primary metric: **AP stays**; Hits@K is secondary. This recommendation was written after viewing the 2024→2025 results, and it is recorded there.
- Document recovery policy: **pending Jack**.
- The FY2026 data do not exist until after 2027-04-30.

## 局限 / 未验证
- Industry and market are current values (未验证).
- Edges for years other than 2024 are not document-verified (未验证).
- A2 only fixes listed-status tags and footnote tags. Other unresolved names remain (未验证).
- Survivor bias. For firms listed later, a first appearance may only mean first observed.
- Absolute AP is about 1–5%, and the intervals are wide.
- After A2, the validation tie between AA and Jaccard decides the best heuristic. Because they are so close, this choice is not robust.
