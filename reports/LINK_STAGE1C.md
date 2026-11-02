# 链路预测 Stage 1C 报告（2026-11-03 KST）

**M3 has not been started.**

**Exploratory results only.** The 2024→2025 test was already viewed in the interim step and in Stage 1B. **A4 (making every year's graph construction consistent) and the within-group metrics were both decided after viewing those results.** So every test number in this report is exploratory. None of them confirms or rejects S1 or S2. Test numbers appear only once per variant.

## 1. A4：全部年份只用 API 规则（data_a2_apionly）
- Rules: A2 name fix, current industry and market (A3), no document recovery.
- Model fitting: on the train years only.
- Model selection: on validation AP (rule unchanged).
- Validation choices: M1 C=0.01; M1+scores C=0.01; **best heuristic Jaccard** (validation AP .0467; AA .0386).

**Positive counts:**

| Split | All | Both connected | One isolated | Both isolated | Non-fin ≥5% (both / one / iso) |
|---|---|---|---|---|---|
| 21→22 train | 260 | 132 | 96 | 32 | 40 (9 / 22 / 9) |
| 22→23 train | 290 | 142 | 113 | 35 | 59 (19 / 30 / 10) |
| 23→24 validation | **303** | 130 | 132 | 41 | 99 (13 / 58 / 28) |
| 24→25 test | 339 | 239 | 84 | 16 (<30, not informative) | 40 (13 / 22 / 5) |

Validation drops from 368 to 303 because the 65 positives that existed only through document recovery are gone. Test and training counts are unchanged.

**Pooled AP on test (exploratory), 1B (A2, with 2024 document recovery) vs 1C (A4):**

| Cell | Metric | 1B | 1C (A4) |
|---|---|---|---|
| All new edges | M1 | .0427 | .0449 |
| | M1+scores | .0469 | .0453 |
| | Best heuristic | AA .0143 | Jaccard .0127 |
| | M1+scores − M1 | +.0042 [−.0053, +.0184] | +.0004 [−.0097, +.0097] |
| | Best heuristic − M1 | −.0285 [−.0605, −.0037] | −.0322 [−.0633, −.0068] |
| Non-fin ≥5% (40) | M1+scores − M1 | −.0031 [−.0129, +.0057] | −.0020 [−.0163, +.0027] |
| | Best heuristic − M1 | −.0128 [−.048, −.0012] | −.0095 [−.0438, −.0007] |
| Both connected | M1+scores − M1 | +.0015 [−.0133, +.0247] | −.0052 [−.0224, +.0133] |
| | Best heuristic − M1 | −.0381 [−.0765, −.0026] | −.0465 [−.0872, −.0083] |
| One isolated (84) | M1+scores − M1 | | +.0004 [−.0021, +.0029] |
| | Heuristics | | all 0 (cannot rank these pairs) |

Other test AP values in 1C: PA .0285, CN .0125, AA .0126, random .0106.

**Full ranking (139 investors, Hits@10 / Hits@50 / MRR):**

| Model | Hits@10 | Hits@50 | MRR |
|---|---|---|---|
| M1 | .036 | .151 | .029 |
| M1+scores | .050 | .194 | .020 |
| Jaccard | .050 | .072 | .016 |
| PA | .058 | .101 | .050 |

## 2. 组内指标（事后分析）
**These metrics were added after viewing the 2024→2025 test results.** Each positive is ranked among its own 101 candidates; ties use expected rank. The intervals come from 1000 bootstrap resamples clustered by investor.

**Test within-group means (AUC / MRR / Hits@10):**

| Cell | M1 | M1+scores | Jaccard | PA | Random |
|---|---|---|---|---|---|
| All (339) | .723 / .215 / .404 | .735 / .221 / .448 | .519 / .048 / .044 | .691 / .167 / .319 | .508 / .055 / .100 |
| Non-fin ≥5% (40) | .558 / .105 / .200 | .549 / .089 / .175 | .498 / .020 / .000 | .441 / .046 / .075 | |
| Both connected (239) | .820 / .274 / .519 | .849 / .288 / .590 | .527 / .060 / .063 | .843 / .230 / .452 | |
| One isolated (84) | .514 / .077 / .131 | .479 / .063 / .107 | .498 / .020 / .000 | | |
| Both isolated (16) | not informative | | | | |

**Test differences (AUC / MRR / Hits@10 [95% CI]):**

| Cell | Comparison | AUC | MRR | Hits@10 |
|---|---|---|---|---|
| All | M1+scores − M1 | **+.012 [+.005, +.020]** | +.007 [−.007, +.027] | +.044 [+.013, +.070] |
| | Jaccard − M1 | −.204 [−.291, −.103] | −.167 [−.263, −.058] | −.360 [−.512, −.189] |
| | M1+scores − Jaccard | +.216 [+.115, +.302] | +.174 [+.077, +.259] | +.404 [+.223, +.563] |
| Non-fin ≥5% | M1+scores − M1 | −.009 [−.030, +.013] | −.017 [−.056, +.006] | −.025 [−.079, .000] |
| | Jaccard − M1 | −.060 [−.151, +.026] | −.086 [−.171, −.025] | −.200 [−.325, −.089] |
| | M1+scores − Jaccard | +.051 [−.036, +.139] | +.069 [+.021, +.130] | +.175 [+.073, +.293] |
| Both connected | M1+scores − M1 | +.030 [+.019, +.047] | +.014 [−.006, +.050] | +.071 [+.037, +.116] |
| | Jaccard − M1 | −.292 [−.361, −.189] | | |
| One isolated | M1+scores − M1 | −.035 [−.046, −.024] | | |
| | Jaccard − M1 | −.016 [−.083, +.051] | | |

**Validation (secondary):**
- All 303 positives: M1 within-group AUC .668, M1+scores .675 (Δ +.007 [−.004, +.019]), Jaccard .541, PA .552.

**Why Jaccard's within-group AUC is so low:**
- About half the candidates have no common neighbours with the investor. Those score 0 and tie, so their AUC is about 0.5.
- PA (preferential attachment) is much better within groups, but it was not selected on validation AP.
- If selection used within-group AUC, the best heuristic on validation would be PA (.552), not Jaccard (.541). That would be a different model choice. It was not executed here.

## 3. S1 / S2 状态（探索性）
- **Pooled AP:**
  - M1+scores − M1 includes 0.
  - Best heuristic − M1 lies entirely below 0, meaning the heuristic is significantly worse than M1.
- **Within-group AUC** (post-hoc; the new primary metric in protocol_confirm):
  - M1+scores − M1 is +.012 [+.005, +.020], which excludes 0.
  - The gain comes from "both connected" (+.030). In "one isolated" it is negative (−.035).
  - Best heuristic − M1 is significantly negative.
- **S2:** not evaluated, because M3 has not been run.
- None of this confirms or rejects S1. The confirmatory test is FY2025→2026, in 2027.

## 4. 其他
- **protocol_confirm.md:**
  - Primary metric: within-group AUC. Secondary: pooled AP.
  - S1/S2 were reworded to use within-group AUC, and validation selection also uses within-group AUC.
  - It records that all of this was decided after viewing the 2024→2025 results.
  - "Jack" was replaced with Mingsen Zhang.
  - The document-recovery policy is fixed as (a).
  - Note: this file is not frozen yet. It has not been reviewed and committed as frozen; the commit time below is when the edits were made.
- **protocol.md:** amendments A4 and A5 were appended, plus a note on the name "Jack" and a column note. The original text is unchanged.
- **Column rename:** `label_doc_only` is now `also_in_2024_doc_set`. It only marks that a pair is also in the 2024 document-recovered set; **it does not mean the label is contaminated**. All 4 data directories were regenerated, and apart from the renamed column the outputs are identical.
- **Git:** author Mingsen Zhang, still with the placeholder email `mingsen-zhang@users.noreply.invalid`. The message contained only "[你选的邮箱]", so no real address was supplied. Nothing was pushed.

## 局限 / 未验证
- Industry and market are current values (未验证).
- No year is document-verified, so API matching misses some edges (未验证).
- A2 only fixes listed-status tags and footnote tags.
- Survivor bias.
- 2024→2025 has been viewed several times, and both the metric and the data rules were changed after viewing. These results cannot serve as confirmation.
- The non-financial ≥5% subset has only 40 test positives, so its intervals are wide.
