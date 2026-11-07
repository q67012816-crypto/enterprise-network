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
- **2026-11-08 01:54 (KST)，修订 A8：度偏差论文的确认性检验（FY2025→26）。Mingsen Zhang 确认冻结（2026-11-08，经分支会话转达，回复“1”）。**
  - 代码提交：`3e7c1cbd7e31183594c292ebbeef9be85f53c8ef`（分支 `claude/v3-dev-interlocks`，标签 `protocol-frozen-A8`）。运行环境：虚拟环境 `D:\claude data\venvs\enet`。运行脚本：`linkpred/src/paper_confirm.py`。
  - 下面第 1–9 节原样复制自修订 A8 草稿（`D:\claude data\paper\protocol_confirm_A8_draft.md`；冻结时的副本：`reports/A8_draft_frozen_20261108.md`）。
  - 追加文本的 SHA-256（`<!-- A8 BEGIN -->` 下一行起、`<!-- A8 END -->` 之前，UTF-8，LF 换行）：`fe963407c992087bb94b4a9c78d98dc8af5e3c474250291960b3564c4c9c68f0`
  - A8 草稿文件冻结时的 SHA-256（UTF-8，LF 换行）：`a7b83f23c48a23129712d8d6da600704a18fc33fed74ed73ab8727d718cf65ed`
  - 本修订不改变上面 M0–M3 研究的规定。2024→25 只作验证年。冻结后不再修改，只能再追加带日期的修订。

<!-- A8 BEGIN -->
## 1. 与现有 protocol_confirm.md 的关系
- 现有 `protocol_confirm.md` 预注册的是 M0–M3 研究（S1/S2：图方法相对 M1、GNN 相对最强启发式）。**它保持不变，照常执行。**
- 本修订**新增**一个独立的确认性研究：度偏差论文的 RQ2（度偏差）与 RQ3（DFGT、结构以外的信息）。两者共用同一批 FY2025→26 原始数据，但数据版本、划分和模型不同，结果分开报告。
- 现有文件中“数据、主指标、测试评估规则”三节的规定，凡本修订没有另行规定的，一律沿用。

## 2. 数据规则
**2.1 数据版本：v3（`build_v3.py`），起始年 2019（即 `data_v3x` 的口径），新增 FY2025→26。**
- 收集：2027-04-30 之后，用相同的 API 调用收集 FY2026 사업보고서、出资行、财务和提交列表（调用次数记入 `data/state/calls.json`）。
- 合格性、持有条件、唯一匹配（精确 / 规范化 / 股票简称）、A2 尾部标记规则：与 `protocol_confirm.md` 相同。
- **v3 的三处修改全部开启：**
  1. `--names-any yes`：曾用名按“t 年当时有效的名称”解析；改名公告只用于确认身份，不带入业务信息；投资行本身仍须在 C_t 前披露。曾用名必须唯一、不与其他公司现名冲突，不做模糊匹配。
  2. `--a2b yes`：去掉不带括号的脚注标记（“空白 + 星号 + 数字”结尾）。
  3. `--newness yes`：首次出现的点对必须满足下列之一：t+1 年出资行期初未持有（期初数量、账面价值、比例都为 0 或空）；或首次取得日在 t+1 财年内。任一出资行显示“期初持有且首次取得早于本财年（或未知）”的，判为早已存在，不作正例也不作负例，写入 `preexisting_pairs.csv`。
- **所有年份只用 API 规则**，不做原始文档补边（选项 (a)）。
- **特征图 G_t：** 只用提交日 ≤ C_t 的出资行。**历史集合 H_t：** 只用于标签。
- **行业和市场：** 当前值，在确认性运行时快照固定；同时报告不含行业和市场的敏感性分析。
- **负例：** 以投资方为锚，每个正例 100 个，排除 H_t ∪ G_{t+1}。2021 年起用种子 20261103 的随机流，2019–2020 用 20261103 + 1000 的独立随机流。
  - **负例随机流（2026-11-08 已核实）：** `build_v3.py` 中 2021 年起各年共用一个随机流，按 FY 顺序、对排好序的正例和候选逐个抽取，所以把 2025 加在最后不会改变 2019–2024 的负例。实测：用当前代码重建 `data_v3x`（`--first-year 2019`，输出到临时目录），`pairs.csv`、`preexisting_pairs.csv`、`nodes.csv`、`eligible.csv`、`graph_2019`–`graph_2025` 与 `data_v3x` 逐字节一致；`check_v3.py` 的 10 项检查全部通过，`unit_checks.csv` 和 `summary.json` 也逐字节一致。`history_undirected.csv` 只有行顺序不同（行顺序来自 Python 集合迭代，随进程的字符串哈希种子变化），点对集合完全相同；打分不读这个文件的行顺序。
  - **回归门槛（2027 年构建后、任何打分之前）：** 运行 `src/check_v3_regress.py --new <2027 年的数据目录> --ref data_v3x --max-fy 2024`，必须输出 `REGRESSION CHECK PASSED`：fy ≤ 2024 的点对、早已存在点对、节点表逐行一致，fy ≤ 2025 的合格名单和特征图逐字节一致，fy ≤ 2024 的历史点对集合一致。不通过就不打分，按第 2.4 节处理。
  - **注意：** `build_v3.py` 输出中的 `split` 列（2023 = 'validation'、2024 = 'test'）是旧研究（M0–M3）的标签，与本修订的划分不同。本修订的打分只按 `fy` 选年份，不读 `split` 列。
- **单元检查：** `check_v3.py` 的 10 项检查必须全部通过，才能进入打分。

**2.2 投资组合与细行业特征（第 1、2 项；`src/pair_feats.py` 的定义）**
- 只用投资方 FY t 出资表中 C_t 前披露、期末持有 > 0 的行；计算时排除候选企业本身那一行。
- port（第 1 项）：`pf_has, pf_size_gap, pf_size_z, pf_prof_gap, pf_loss_gap, pf_simple_x_size, pf_simple_x_Y, pf_mgmt_x_size, pf_n_x_size`。
- ind（第 2 项及行业历史亲和度）：`same3, same4, same5, hist_has, aff2, aff3, aff4, rev_has, rev2, rev3`。
  - 注意：其中 `hist_has / aff* / rev*` 依赖过去的持股关系（G_s，s ≤ t），本质是结构信息。开发阶段 DFGT 的“第 1、2 项”变体（`pf=('port','ind')`）**包含**这一组。
- **已定（2026-11-03，Mingsen Zhang，经分支会话转达，原话：“好的我同意”）：GNN 的兼任检验改用不含结构信息的打包 C3′**，即 port + fine（`same3, same4, same5`），**不含** `hist_has / aff* / rev*`（变体 `dfgt+port+fine` 与 `dfgt+port+fine+il`，见 5.3 的 M_f′、M_g′）。原打包（port + ind 全部，`dfgt+pf`、`dfgt+pf+il`）只作敏感性报告。

**2.4 数据异常的处理（事先写定，不做临时修补）**
- 如果 2027 年 `exctvSttus`（兼任）或 `hyslrSttus`（最大股东）接口不可用、字段格式改变，或 FY2024–2025 的这两类数据无法按第 3 节的规则下载：不改特征定义、不换数据源。依赖这类数据的检验（兼任：P1、S_iso、C3′、C4、S3 和兼任相关的敏感性；最大股东：C5）报告为“不可解读”，并说明原因。其余检验照常判定。
- **覆盖率门槛：** 测试年全部点对（正例和负例）中 `il_has_a = 1` 且 `il_has_o = 1` 的比例，若低于三个开发验证年（2021–2023）最低值的一半，同样把依赖兼任的检验报告为“不可解读”；最大股东用 `hs_has_a`、`hs_has_o` 按同样规则处理。开发年的比例（2026-11-08 计算）：兼任 2021 .643、2022 .767、2023 .868，所以门槛 = .643 / 2 ≈ **.32**；最大股东 2021 .638、2022 .760、2023 .861，门槛 ≈ **.32**。验证年 2024 和测试年 2025 的比例在确认运行中计算，无论高低都在结果中报告。
- 第 2.1 节的回归门槛或 `check_v3.py` 的单元检查不通过时，不打分。原因查明后的任何修改都作为新的修订记录，并在结果中标出。
- 已定（2026-11-08，Mingsen Zhang，经分支会话转达，对“确认，现在冻结”回复“1”）：本节规则、运行环境 enet、代码提交 3e7c1cb 一并确认。

## 3. 兼任特征的定义（第 3 项；`src/interlock_feats.py` 的定义）
**3.1 数据：** OpenDART `exctvSttus`（年报“임원현황”，含등기임원和미등기임원）。FY2024 和 FY2025 在 2027-04-30 之后与 FY2026 数据一起下载（已定（2026-11-03，Mingsen Zhang，经分支会话转达）：FY2024 兼任数据到 2027 年与 FY2026 一起下载）。
**3.2 同一个人：** 姓名（去掉所有空白）+ 出生年月（YYYYMM，6 位）。缺出生年月或姓名的行丢弃；只用 api_status = 000 的行。
**3.3 as-of 规则（主分析）：** FY t 只用 `rcept_no` 日期 ≤ C_t（t+1 年 4 月 30 日）的年报。API 只返回最新版本，所以 C_t 之后提交的更正年报整份剔除。
- 敏感性分析（`--lenient`）：保留这些更正年报。只作敏感性报告，不进入成功标准。
**3.4 执行人员：** `rgist_exctv_at` 和 `ofcps` 任一字段含“사외”或“감사”的，不算执行人员；其余（包括미등기임원）都算。各年同一口径。
**3.5 点对特征（a = 投资方，o = 候选企业）：**
| 特征 | 定义 |
|---|---|
| `il_direct` | FY t 两家公司共有的人数，log1p |
| `il_inside` | 其中至少在一家公司是执行人员的人数，log1p |
| `il_2hop` | FY t 与 a、o 都共有人员的第三家公司数，log1p |
| `il_hist` | 在 s ≤ t 的任一年在 a 任职、且在 s′ ≤ t 的任一年在 o 任职的人数，去掉 FY t 的直接兼任，log1p |
| `il_has_a` / `il_has_o` | 该公司 FY t 是否有可用的임원현황（覆盖指示） |
**3.6 兼任边（用于消息传递）：** FY t 中两家合格企业共有至少一人，就连一条无向边，并入 G_t。度分组仍只按持股图 G_t 计算。`ile_in` 版本只保留涉及执行人员的边。

**3.7 最大股东特征（第 4 项；`src/hyslr_feats.py` 的定义；只作次要报告，见 7.2）**
- 数据：OpenDART `hyslrSttus`（年报“최대주주 및 특수관계인의 주식소유 현황”）。FY2024、FY2025 与 FY2026 数据一起在 2027-04-30 之后下载。只用 api_status = 000 的行；“계”等小计、自己股份、员工持股组合等通用条目剔除。
- as-of：同 3.3，FY t 只用 `rcept_no` 日期 ≤ C_t 的年报（宽松版只作敏感性）。
- 名称：法人名称用与出资表相同的归一化键；自然人只用姓名（去空白）。某年出现在**超过 5 家**公司名单中的名称不参与匹配（事先设定，不调）。
- “最大股东本人”：`relate`（去空白）含“본인”，或者正好是“최대주주”/“최대주주(본인)”。
- 点对特征：`hs_same_max`（双方最大股东本人是同一主体）、`hs_shared`（共有名称数，log1p）、`hs_corp`（其中法人名称数，log1p）、`hs_exec_ao` / `hs_exec_oa`（一方自然人股东是另一方当年董事或高管的人数，同名匹配，log1p）、`hs_has_a` / `hs_has_o`（覆盖指示）。

## 4. 划分
- **训练：** 2019→20、2020→21、2021→22、2022→23、2023→24。
- **验证（只用于早停和选择）：** 2024→25。
- **确认性测试：** 2025→26，只评估一次。
- **已定（2026-11-03，Mingsen Zhang，经分支会话转达）：训练包含 2019–2020**，与开发阶段一致（早期出资表覆盖较差，作为限制报告）。

## 5. 模型（全部固定，不再调参）
**5.1 超参数：** 沿用 A7 在验证集上选出、开发阶段一直使用的值：
- M1 / M1+特征：逻辑回归，C = 0.1，标准化，lbfgs；
- GraphSAGE：均值聚合，2 层，hidden 128，lr 1e-2，dropout 0.3，MLP 解码器；
- DFGT：lr 3e-3，λ 0.1，hidden 64，2 层，4 头，k-means M = 5，2 跳上下文，结构注意力 3 跳；
- 无图对照：DFGT 架构，所有边去掉（只剩自环）；
- GNN 早停：验证组内 AUC，patience 15，最多 150 轮。
- **已定（2026-11-03，Mingsen Zhang，经分支会话转达）：不在 2024→25 上重新调参**，以免确认时的模型与开发阶段证据不一致。

**5.2 种子：** 每个 GNN 变体跑 10 个种子（1–10）。测试分数 = 10 个种子分数的平均。同时保存每个种子单独的分数，用于第 7.3 节的种子配对分析。

**5.3 参与确认的模型（开发阶段的名称 → `paper_dev.py` 变体名）：**
| 编号 | 模型 | 变体名 |
|---|---|---|
| M_a | M1 | `M1` |
| M_b | M1 + 兼任 | `M1+il` |
| M_c | GraphSAGE | `sage` |
| M_d | 无图对照 | `dfgt-nograph` |
| M_e | DFGT | `dfgt` |
| M_f | DFGT + 第 1、2 项 | `dfgt+pf` |
| M_g | DFGT + 第 1、2 项 + 兼任 | `dfgt+pf+il` |
| M_f′ | DFGT + 第 1 项 + 细行业（不含 hist/aff/rev 结构特征） | `dfgt+port+fine` |
| M_g′ | M_f′ + 兼任 | `dfgt+port+fine+il` |
| M_h | DFGT + 兼任边 + 第 1、2 项 + 兼任（完整模型） | `dfgt+ile+pf+il` |
| M_x1 | M1 + 最大股东（第 4 项，次要报告） | `M1+hs` |
| M_x2 | M1 + 兼任 + 最大股东（第 4 项，次要报告） | `M1+il+hs` |
| M_x3 | M_g′ + 最大股东（第 4 项，次要报告） | `dfgt+port+fine+il+hs` |

- 第 4 项不改变最终模型：M_h、P1 和第 7 节的其他检验都不变（已定（2026-11-06，Mingsen Zhang，本会话中用户本人选择“Secondary, report only”））。
- **运行脚本：** `src/paper_confirm.py`（2026-11-08 写成）。它覆盖上表全部 13 个模型，另加第 7.2 节报告项需要的 3 个变体：`dfgt+ile`（S3）、`M1+il-lenient`（兼任宽松版）、`M1-noindmkt`（M1 不含行业和市场；新加的变体，开发阶段没有跑过，只用于敏感性报告）。训练 2019–2023、验证年 2024 早停、测试年 2025 只打分一次；GNN 10 个种子取平均并另存每个种子的分数；输出已存在时拒绝运行；每个模型先写验证分数并记录 SHA256，再写测试分数并记录 SHA256；`manifest.json` 记录代码提交、源文件、数据文件、特征文件的 SHA256、运行环境，以及第 2.4 节的覆盖率和门槛判定。脚本在正式模式下要求测试年 = 2025，任何模式下拒绝把 2024 作测试年。
- **演练（2026-11-08）：** 只用开发年（训练 2019–2021、验证 2022、“测试” 2023），`--dry-run`，输出 `linkpred/out_confirm_dryrun/`。演练只检查流程，结果不进入任何检验或报告。
  - 演练的验证 AUC 与开发阶段同年份的折（F2：训练 2019–2021、验证 2022）一致：M1+il、DFGT 相同，其余相差 ≤ .0007（来自 scikit-learn 版本和 GPU 噪声）。两条评估命令都在演练输出上跑通。
  - GPU 上固定种子的 GNN 不能逐位复现：同一提交跑两次，DFGT 的验证 AUC 为 .7391 和 .7372。确认测试只运行一次，以那一次的结果为准。
- **实现冻结：** 冻结时记录以下文件所在的提交哈希：`paper_confirm.py`、`paper_dev.py`、`paper_models.py`、`dev_models.py`、`paper_data.py`、`pair_feats.py`、`interlock_feats.py`、`hyslr_feats.py`、`build_v3.py`、`check_v3.py`、`check_v3_regress.py`。冻结提交：`3e7c1cbd7e31183594c292ebbeef9be85f53c8ef`（2026-11-08，`claude/v3-dev-interlocks` 分支；冻结时核对过 `linkpred/src` 自该提交以来没有变化；演练的 `manifest.json` 记录了这个提交和各源文件的 SHA256）。确认时用同一提交运行。
  - 冻结后只允许改年份常量和路径，并在结果报告中列出 diff。预计 2027 年只需改：`build_v3.py` 的 `FY`（到 2025）和 `split` 字典（加入 2025 一项，标签任意，打分不读）；`pair_feats.py`、`interlock_feats.py`、`hyslr_feats.py` 的 `YEARS`（到 2025）；`paper_dev.py` 的 `GRAPH_YEARS`（到 2025，评估时用）；以及 `--data`、`--feat`、`--out` 路径。`paper_confirm.py` 的年份已按正式设置写好，运行时自动设置 `GRAPH_YEARS` 和 `PF_ZERO_FROM`。
  - 18f7bbe 之后加入的代码（DCM、新基线、SEAL、边丢弃、`--save`、确认用的测试年打分）都是新增分支或带默认值的参数：不打开这些选项时，第 5.3 节的变体与之前完全一样。2026-11-08 实测：修改后的 `paper_dev.py` 与 HEAD、18f7bbe、0f73e93 三个版本跑 `M1` 的输出逐字节一致。
- **运行环境（冻结）：** Python 虚拟环境 `D:\claude data\venvs\enet`（numpy 2.5.4、pandas 3.1.0、scikit-learn 1.9.1、scipy 1.18.1、torch + CUDA、PyG 2.8.1；准确版本以演练 `manifest.json` 的 `environment` 为准）。全部模型在同一环境、同一进程中运行。
  - 说明：开发阶段逻辑回归的分数（包括 P1 的开发证据）当时是用系统 Python（scikit-learn 1.8.0、numpy 2.4.4、pandas 3.0.3）跑的；GNN 用上面的虚拟环境。同一代码在虚拟环境中重跑 `M1`、`M1+il`，单个分数最多差 .013，但 P1 的开发估计不变：低连接组 +.034，单侧 95% 下限 +.021，双侧 [+.019, +.052]（B = 10,000）。

## 6. 指标与分组
- **主指标：** 组内 AUC（每个正例在自己的 101 个候选中排序，并列按期望秩，对正例取平均）。
- **次要指标：** 组内 MRR、Hits@10、混合池 AP。
- **度分组（按持股图 G_t，无向）：**
  - 低连接：新边两端中较小的度 ≤ 1；
  - 高连接：两端的度都 ≥ 2；
  - 孤立：较小一端的度 = 0（低连接的子集）。
- **度偏差差距：** Δ = AUC(高连接) − AUC(低连接)。
- **最小样本：** 测试年某组少于 30 个正例，报告为“不足以解读”，不判定该组的标准。
- **区间：** 按投资方聚类的配对 bootstrap，`numpy.default_rng(20261103)`，95% 百分位区间。**抽样次数：主检验 P1 和次要确认性假设 C1、C2 用 B = 10,000；其余检验用 1,000。**（已定（2026-11-03，Mingsen Zhang，经分支会话转达，原话：“好的”）：P1 也用 10,000，原因是 P1 的下界若落在 0 附近，1,000 次抽样的随机误差可能影响判定。）评估命令：`paper_dev.py eval --out out_confirm --boot 10000 --only "M1+il:M1,dfgt:M1" --gaps dfgt --save confirm_eval_boot10000.json`；其余检验：`paper_dev.py eval --out out_confirm --boot 1000 --save confirm_eval_boot1000.json`。
- **种子配对（辅助）：** 每个种子单独评估，按同一种子配对求差，报告均值 ± 标准差和 10 个种子中为正的个数。只作辅助报告，不进入成功标准。

## 7. 成功标准（确认性）
> 所有判定都在 2025→26 测试年上做，只用这一个测试年（见第 8 节）。已定（2026-11-03，Mingsen Zhang，经分支会话转达，原话：“好的我同意”）。

**7.1 主检验（唯一）**
| 编号 | 检验内容 | 比较 | 组 | 方向与判定 | 开发阶段证据 |
|---|---|---|---|---|---|
| **P1（原 S2）** | 兼任信息帮助低连接企业（线性模型） | M_b − M_a，即 `M1+il` − `M1` | **低连接**（较小一端的度 ≤ 1） | **单侧 95%**：配对聚类 bootstrap（**B = 10,000**，种子 20261103）差值分布的 **5% 分位 > 0** 即“满足”；同时报告双侧 95% 区间 | +.034，单侧 95% 下限 +.021；双侧 95% [+.019, +.052]（B = 10,000；逻辑回归，确定性，三个验证年合并，372 个低连接正例）；单年功效 .927（单侧），见第 8 节 |

- 只有一个主检验，所以**不做多重比较校正**。
- 开发阶段的证据来自三个滚动验证折（分别用 2–4 个训练年，各自打分后合并）；确认运行只训练一次，用 5 个训练年（2019–2023）、在 2024 早停、给 2025 打分。两者的训练设定不完全相同。
- 论文关于“结构以外的信息能帮助低连接企业”的核心结论，只以 P1 为依据。

**7.2 次要检验**（C1、C2 保留判定；其余只报告点估计和 95% 区间）
- **最重要的次要检验 S_iso：** M_b − M_a 在**孤立组**（较小一端的度 = 0）。报告点估计和区间（单侧 95% 下限，同时报告双侧 95% 区间）。开发阶段 +.034，单侧 95% 下限 +.019，双侧 [+.017, +.054]（B = 10,000，与 P1 同一套抽样；248 个孤立正例），单年功效约 .81（单侧）。**预先写明：区间含 0 可能来自功效不足，不等于兼任对孤立企业没有效果。**
- **GNN 版兼任检验（C3′，不作判定）：** M_g′ − M_f′（`dfgt+port+fine+il` − `dfgt+port+fine`）在孤立组、低连接组、全部组的点估计和区间。开发阶段：孤立 +.025 [+.011, +.043]，低连接 +.021 [+.007, +.037]，全部 +.013 [+.003, +.025]。
- **完整模型（C4，不作判定）：** M_h − M_e 在全部、低连接、孤立组。开发阶段：全部 +.017 [+.004, +.031]，低连接 +.016 [−.002, +.035]，孤立 +.007 [−.017, +.030]。
- **次要确认性假设 C1、C2（保留“满足/不满足”判定）。** 已定（2026-11-03，Mingsen Zhang，经分支会话转达，原话：“同意”）：
  - C1：DFGT（M_e）的 Δ = AUC(高连接) − AUC(低连接) > 0。开发阶段 Δ = .200 [.160, .237]（B = 10,000）。
  - C2：M_e − M_a > 0，在全部组和低连接组**各自单独判定**。开发阶段（B = 10,000）：全部 +.074 [+.051, +.097]，低连接 +.090 [+.058, +.121]。开发阶段按本节规则的 Holm 判定：C1、C2 全部、C2 低连接都“满足”（3 个 p 均 < 1/B = .0001）。
  - **抽样次数：** C1、C2 族的区间和 p 值都用 **B = 10,000** 次配对聚类 bootstrap（按投资方聚类，`numpy.default_rng(20261103)`），在确认运行前写死，不再更改。B = 10,000 时 p 值的分辨率是 2/B = .0002，远小于 Holm 的最严阈值 .05/3 ≈ .0167。（P1 同样用 10,000 次；其余检验按第 6 节用 1,000 次。）
  - **族内多重比较校正：C1、C2 族内用 Holm 校正，与 P1 分开。** 这个族共 3 个假设：C1、C2（全部组）、C2（低连接组）。
    - p 值：双侧 bootstrap p 值 = 2 × min（差值 ≤ 0 的抽样比例，差值 ≥ 0 的抽样比例），上限为 1；C1 用 Δ 本身。
    - Holm 步骤：把 3 个 p 值从小到大排序，依次与 .05/3、.05/2、.05/1 比较；某一步 p 不小于对应阈值，就在这里停止，它和之后的假设都不显著。
  - **判定为“满足”，需 Holm 校正后显著，且点估计方向与假设一致**（C1：Δ > 0；C2：M_e − M_a > 0）。校正后显著但方向相反的（例如 DFGT 反而比 M1 差），判定为**“不满足（方向相反）”**，不算满足。Holm 的逐步比较只看 p 值，方向只影响最后的判定。
  - P1 不在这个族里，按 7.1 单独判定（Mingsen Zhang，2026-11-03，用户本人在本会话中指示）。
  - **预先写明：** C1 支撑论文的 RQ2（度偏差），C2 支撑 RQ3 前半部分（消融和原因分解），**但 P1 仍是唯一决定核心结论的主检验**。（措辞已定，2026-11-03，Mingsen Zhang，经分支会话转达，原话：“同意”。）C1、C2 是否满足，不改变对 P1 的解释。
- S1：图结构只帮助有连接的企业。M_e − M_d 在高连接组的差值；孤立组只报告区间，**不作“无差异”判定**。
- S3：兼任作为边。`dfgt+ile` − M_e，只报告。
- **敏感性：** 原 C3 打包（M_g − M_f，含 `hist_has/aff*/rev*`；开发阶段孤立 +.036 [+.012, +.058]）；兼任宽松版（保留 C_t 后的更正年报）；M1 不含行业和市场。
- **透明说明（冻结前的额外探索）：** 冻结前，在开发阶段（只用训练年和验证年）还尝试过两类信息：高管履历中的前任职公司，以及 5% 以上大量持股人。结果见 `reports/DEV_RESULTS.md` 第 13 节（探索性）。履历信息在兼任之上没有稳健增量（低连接组 LR +.007 [−.000, +.016]，DFGT +.004 [−.002, +.012]）；5% 持股人接口只返回最近约两年的报告，开发年份无可用数据。两者都**不进入**本修订的任何检验或报告项。
  - 此外，冻结前在开发阶段还做了以下分析，全部只用训练年和验证年（或公开基准数据），都**不进入**本修订的任何检验或报告项：`DEV_RESULTS.md` 第 14 节（双通道模型 DCM，无效）、第 15 节（命题 4 的增益上界）、第 16–17 节（TNSE 补充实验 A：公开基准上的受控孤立节点；实验 B：GAE、Cold Brew、DegFairGNN、Tail-GNN 四个新基线）、第 18 节（追加实验 C1–C3：训练时的边丢弃、新基线跑在公开基准上、SEAL）。
- **C5（第 4 项，最大股东）：只作次要报告，不作判定。** 已定（2026-11-06，Mingsen Zhang，本会话中用户本人选择“Secondary, report only”）。报告以下三项的点估计和 95% 区间（1,000 次 bootstrap），低连接、孤立、全部三组：
  - M_x1 − M_a（M1 + 最大股东 − M1，不含兼任）。开发阶段：全部 +.013 [+.006, +.021]，低连接 +.015 [+.006, +.025]，孤立 +.011 [+.002, +.021]。
  - M_x2 − M_b（在已有兼任的前提下再加最大股东）。开发阶段：低连接 −.001 [−.003, +.002]，孤立 −.001 [−.004, +.001]。
  - M_x3 − M_g′（GNN，C3′ 打包下再加最大股东）。开发阶段：低连接 −.000 [−.007, +.007]，孤立 −.003 [−.011, +.005]。
  - 预先写明的解读方向：开发阶段显示“最大股东信息单独有效，但被兼任信息覆盖”（两者高度重叠：有大股东信号的正例 4.6%，有兼任信号的 6.2%，合计 7.6%）。后两项区间含 0 不等于“无效”，只说明在兼任之外没有可检出的增量。

**7.3 多重比较**
- 已定（2026-11-03，Mingsen Zhang，经分支会话转达，原话：“好的我同意”）：P1 是唯一的主检验，不做多重比较校正。
- 已定（2026-11-03，Mingsen Zhang，经分支会话转达，原话：“同意”）：C1、C2 为次要确认性假设，用双侧 95% 判定；C1、C2 族内（3 个假设）用 Holm 校正，与 P1 分开（Holm 一句为 2026-11-03 用户本人在本会话中补充）。第 7.2 节其余次要检验（S_iso、C3′、C4、S1、S3、敏感性）只报告点估计和区间，不作判定。

## 8. 统计功效说明（冻结前必须读）
- 确认只有一个测试年。以 2024→25 为参照，一年约 285 个正例，其中低连接约 140 个，孤立约 78 个【DEV 数据表：2024 年 min_deg ≤ 1 为 140，min_deg = 0 为 78】。
- **模拟（2026-11-03，DEV_RESULTS §11.4，`linkpred/src/power_sim.py`；功效按开发阶段的效应估计计算，可能偏乐观，存在赢家诅咒）：** 以三个验证年的观测效应作为真实效应，按“折 × 投资方”聚类重抽到测试年样本量；每个模拟年内部做 1000 次聚类 bootstrap；重复 1000 次。表中是区间下限 > 0 的比例（单侧 95% / 双侧 95%）。

| 标准 | 组（单年 n） | 开发效应 | 单年：单侧 / 双侧 | 加入 FY2026→27（两年）：单侧 / 双侧 |
|---|---|---|---|---|
| C1 | 全部（285） | Δ = .200 | 1.00 / 1.00 | 1.00 / 1.00 |
| C2 | 全部（285） | +.074 | .997 / .994 | — |
| C2 | 低连接（140） | +.090 | .966 / .946 | — |
| C3（M_g − M_f） | 孤立（78） | +.036 | .570 / .465 | .783 / .691 |
| C3 | 低连接（140） | +.027 | .612 / .503 | .838 / .745 |
| C3 | 全部（285） | +.017 | .715 / .596 | .916 / .855 |
| C3′（M_g′ − M_f′，不含结构信息，次要） | 孤立（78） | +.025 | .626 / .494 | .872 / .783 |
| C3′ | 低连接（140） | +.021 | .589 / .442 | .829 / .740 |
| C4 | 全部（285） | +.017 | .570 / .464 | .807 / .722 |
| C4 | 低连接（140） | +.016 | .275 / .199 | .433 / .314 |
| S_iso（M_b − M_a，逻辑回归；最重要的次要检验） | 孤立（78） | +.034 | .810 / .701 | .982 / .955 |
| **P1（主检验）** | **低连接（140）** | +.034 | .927 / .856 | .997 / .992 |

- **结论：**
  - C1、C2 在单一测试年上功效充足。加上族内 Holm 校正后（最严阈值双侧 .05/3 ≈ .017）：C2 全部组 .981，C2 低连接组 .856（同一模拟、双侧 .05 时分别为 .997、.930；与上表的差别是模拟的随机误差，约 ±.015）；C1 的区间远离 0，不受影响。
  - **C3、C3′、C4 在单一测试年上功效只有约 .45–.70**：即使效应真实存在，也有约一半的概率不满足。
  - 逻辑回归版的兼任检验（S2）功效明显更高，因为没有 GNN 的种子方差。
- 注：功效模拟中每个模拟年内部用 1,000 次 bootstrap；确认时 P1 用 10,000 次只减少抽样随机误差，不改变预期功效。
- **限制：** 功效按开发阶段的效应估计计算，**可能偏乐观**：开发阶段的效应是从多个变体中挑出来的，存在赢家诅咒（winner's curse），真实效应往往比开发阶段小，实际功效可能更低；各组单独模拟，未计算多个组同时满足的联合概率；未计入 Holm 校正（校正后功效更低）。
- **已定（2026-11-03，Mingsen Zhang，经分支会话转达，原话：“一年的就可以了”）：确认性检验只用一个测试年 FY2025→26，不加入 FY2026→27。** 因此下面的选项 3 不采用；表中“两年”一列只作参考。
- **已定（2026-11-03，Mingsen Zhang，经分支会话转达，原话：“好的我同意”）：** 主检验 = P1（M1 + 兼任 − M1，低连接组，单侧 95%，单年功效 .927）；孤立组为最重要的次要检验（单年功效约 .81 单侧）；GNN 版 C3′、C4 降为次要，不作判定。下面是当时列出的选项，留作记录：
- 应对选项（记录）：
  1. 照常预注册，事先说明“不满足”可能来自功效不足，不等于效应不存在；
  2. 把 C3 的主组改为低连接组（样本约 140），孤立组降为次要（模拟显示低连接组的功效与孤立组相近，单靠换组改善有限）；
  3. ~~把 FY2026→27（2028 年数据）也写入确认，两年合并判定~~（**不采用**，见上）；
  4. 把兼任的主检验换成逻辑回归版（S2 升为主要标准），DFGT 版 C3 降为次要（单年功效 .70–.93）。**待确认的建议方案：** 主检验 = M1 + 兼任 − M1（S2），低连接组，单侧 95%（单年功效 .927）；同时待定：用 C3′ 替代 C3、GNN 版 C3′ 与 C4 降为次要；
  5. 单侧检验（方向已预先确定为 > 0）：每格功效约提高 .1。
- 第 4 项（C5，只报告）的单年功效（单侧/双侧）：M1 + 最大股东 − M1，全部 .931/.889，低连接 .741/.629，孤立 .448/.301；在已有兼任的前提下再加最大股东，逻辑回归低连接 .018/.006，GNN 低连接 .051/.027，即接近无效应。

## 9. 执行规则
- 测试只运行一次，单条命令（`python src/paper_confirm.py run --data <2027 年的数据目录> --feat <特征目录> --out out_confirm`）；输出目录中已有分数文件或 `manifest.json` 时脚本拒绝运行。
- 运行顺序：数据构建 → 单元检查（`check_v3.py`）→ 回归门槛（`check_v3_regress.py`，第 2.1 节）→ 特征计算（第 2、3 节）和覆盖率检查（第 2.4 节）→ 在训练年训练、验证年早停 → 记录验证分数文件哈希 → 测试年打分一次（记录测试分数文件哈希）→ 评估（第 6 节的两条命令，记录输出文件哈希）。
- 结果出来后不修改任何东西。之后发现的问题只能作为新的修订记录，相关结果标为探索性。
- 本修订由执行者（Claude）根据 Mingsen Zhang 的要求起草；所有【决定】项由 Mingsen Zhang 确定。
<!-- A8 END -->
