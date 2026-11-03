# VERIFY_STAGE2：linkpred Stage 2（M3, GraphSAGE）独立核验

- 核验人：独立核验者（未参与建模），2026-11-03。
- 核验对象：`origin/master` @ `c63217e`。
- 只读核验：仓库里没有修改任何代码、数据或结果，本文件是唯一新增文件。
- 第 1–7 节只用仓库里保存的文件，没有重新训练模型。第 8 节在仓库之外的临时副本里，从 GitHub 材料重建了数据并重训了 M3，其中包括一次核验者自己的测试集打分。
- 重算工具：第 1–7 节用 python 3 + numpy 2.4.4 / pandas 3.0.3 / scikit-learn 1.8.0；第 8 节的环境见该节。完整脚本见附录。
- 判定只有三种：**通过** / **不通过** / **无法验证**。部分可查、部分不可查时，分别写明。

## 总表

| # | 检查项 | 判定 |
|---|---|---|
| 1 | 标签早于所有 M3 输出；之后两份协议只追加 | **通过**（时间线依据的是 commit 时间，它由作者本地时钟决定，见下文） |
| 2 | A6（PA）只依据验证集组内 AUC 选出 | **通过**（选择准则本身是看过测试后才定的，已披露；这次改动让 S2 更难成立） |
| 3a | M3 配置选择只用验证集 | **通过** |
| 3b | 测试集只评估了一次 | **无法验证**（现有证据都一致，找不到反证；但无法排除删除文件后重跑） |
| 4 | 消息传递只用 G_t，没有混入 G_{t+1} 或测试正例 | **通过**（代码和保存的图都查过；用仓库材料重建的 G_t 与提交版逐字节一致，见第 8 节） |
| 5 | 独立重算测试指标和 bootstrap 区间 | **通过**（与报告完全一致，差异小于 1e-15）；但有 3 处“区间不含 0”对 bootstrap 种子敏感 |
| 6 | 报告列出的偏差和未验证项 | 报告第 4 节实际列了 **9 项**，不是 7 项。逐项判定见下文：第 4 项和 A1 的写法矛盾，第 9 项已过时；都不改变主结论 |
| 7 | 5 个种子都完成；测试分数是 5 个种子的平均 | **通过** |

---

## 1. 冻结标签和协议只追加 —— 通过

**证据：**
- `protocol-frozen-pre-m3` 是轻量标签，没有 tagger 时间，直接指向 commit `1aa6762`（2026-11-03 01:13:10 +0900，A6 提交）。
- 标签所在的树里没有任何 M3 文件：`git ls-tree -r protocol-frozen-pre-m3` 中不含 `out_2_m3/`、`m3_sage.py`、`m3_eval.py`。`git grep -i "graphsage|m3_sage|out_2_m3"` 只在 `protocol.md` 的文字里出现。
- 所有 M3 文件只出现在两个 commit：`bc54f22`（01:21:48，代码、验证网格、selection）和 `cb43072`（01:23:55，测试分数和评估）。两者都在标签之后。
- `search.log` 记录总耗时 405 s，最后一行是 `selected ...`。按 01:21:48 提交倒推，搜索大约在 01:14 开始，晚于 01:13:10。这和“标签早于第一个 M3 输出”一致，但间隔不到 1 分钟。
- `git diff --stat protocol-frozen-pre-m3 HEAD -- linkpred/protocol.md linkpred/protocol_confirm.md` 输出为空：标签之后两份文件**没有任何改动**，追加也没有。

**限制：**
- git 的 author/commit 时间由提交者本地时钟决定，可以伪造。能独立证明的只有 commit 的先后关系，证明不了真实的墙钟时间。
- 标签之前的情况（不在本项要求的范围内，仅供参考）：`protocol.md` 从 `285f96a` 到 `1aa6762` 每次都只追加，没有删除任何行。`protocol_confirm.md` 在 `358a1ff` 被**原地改写**：主指标从 AP 改为组内 AUC，S1/S2 的措辞也改了。文件自己的“修订记录”披露了这次改写，并说明是在看过测试之后做的。

## 2. A6：最强启发式按验证组内 AUC 选出 —— 通过

**证据：**
- 用 `out_1c_a4/val_scores.csv` 独立重算。该文件只含 `split == validation`，303 个正例。按 101 个候选、并列取期望秩计算组内 AUC：CN 0.54109、AA 0.54112、Jaccard 0.54109、**PA 0.55193**。与 A6 表格逐位一致，PA 最高，领先 0.011，没有触发并列规则。
- A6 引用的 `within_val_scores.json` 在 `358a1ff`（早于 A6）已经提交，A6 的数字来自已有文件。

**需要说明的背景（不影响判定）：**
- A6 提交时，`out_1c_a4/within_test_scores.json`（同在 `358a1ff`）已经包含测试集组内 AUC：PA 0.691，Jaccard 0.519。也就是说，作出 A6 的人当时**能看到测试结果**。这一点 A6 原文已披露。
- 改准则的方向是**保守的**。原 §5 按验证 AP 选出的是 Jaccard，测试集上 M3 − Jaccard 为：
  - 组内 AUC：+.213 [+.118, +.296]
  - AP：**+.027 [+.006, +.047]**

  也就是说，如果完全按原预注册规则（AP 准则，比较对象 Jaccard），S2 反而会成立。改成 PA 之后 S2 更难成立。所以看过测试后改准则，没有让结论偏向 M3。

## 3a. M3 配置选择只用验证集 —— 通过

**证据：**
- `m3_sage.py search` 分支只用 `TR`（train）和 `VA`（validation）两部分点对（`pair_index(TR)`、`pair_index(VA)`）。测试点对只在 `test` 分支构造。`y_tr` 虽然是全表标签，但只按训练行下标取用。
- 早停用验证 AP，选配置用验证组内 AUC。`val_grid_runs.csv` 只有 config、seed、val_AP、val_within_AUC、best_epoch、epochs_run 六列，没有测试列。
- 我用 `val_grid_runs.csv` 独立重算：32 个配置 × 5 个种子 = 160 行。按 5 个种子的平均验证组内 AUC 排序，最优是 `h128_l2_lr0.01_do0.3_mlp`（0.6926），按验证 AP 排序也是它。这与 `selection.json` 一致。

**附带观察（不是不通过的理由）：**
- 选中的配置在 hidden、层数、lr、dropout 四个维度上都取了网格的上端，网格可能没有覆盖到更优的区域。这只能说明网格设计偏窄，和泄漏无关。
- 本会话收到的原始任务说明写的 M3 网格是“层数 {1,2}、隐藏维度 {32,64}、学习率 {1e-3,1e-2}”。`protocol.md §7`（`285f96a` 预注册）写的是 hidden {64,128} × 层数 × lr × dropout {0,0.3} × 解码器 {dot, MLP}。网格在 M3 之前就已写定，不是事后调整；但它和原始任务说明不一致，协议里也没有标为 [executor choice]。是否得到确认：**未验证**。

## 3b. 测试集只评估一次 —— 无法验证（证据一致，但无法完全证实）

**支持“只评估了一次”的证据：**
- `TEST_DONE.json` 的时间是 01:21:50，只比提交 `selection.json` 的 `bc54f22`（01:21:48）晚 2 秒，选定配置也相同。
- `test` 分支在 `test_scores_m3.csv` 已存在时拒绝运行。
- `test_scores_m3.csv` 只在 `cb43072` 被加入过一次，git 历史里没有更早的版本。
- `search.log` 只有一次完整搜索：160 行，加一行 `selected`。

**无法排除的情况：**
- 防重跑只靠“文件存在”判断，删掉文件就能再跑。
- `models/`（19 MB）没有入库，测试阶段的 stdout 没有保存成日志，`TEST_DONE.json` 的时间来自本地时钟。
- 所以“没有删除后重跑”这一点无法从仓库独立证明。

**另外：** 2024→2025 测试年在 interim、Stage 1B、1C 已经被多次查看。报告开头已经把所有测试数字声明为探索性，这一声明是正确的。

## 4. 消息传递只用 G_t —— 通过

**代码检查（`m3_sage.py`）：**
- 只读取 `graph_2021`…`graph_2024`（`range(2021, 2025)`），**从未读取 `graph_2025.csv`**。
- 每年的邻接矩阵 `A` 只用 `G[t]` 构造。
- 点对按 `fy` 取对应年份的 `x` 和 `A`。`pairs.csv` 中 train 的 fy ∈ {2021, 2022}、validation 为 2023、test 为 2024。所以测试打分用的是 G_2024，验证用的是 G_2023。
- 训练正例（t+1 年的新边）只作为损失的标签，不进入 `A`。
- 特征的截断、填补和标准化统计量只用 2021、2022 两个训练年份的节点。

**数据检查（`data_a2_apionly/`，按无向点对）：**

| t | G_t 边数 | t→t+1 正例 | 正例在 G_t 中 | 任意点对（含负例）在 G_t 中 | 正例在 G_{t+1} 中 | G_{t+1} 新边混入 G_t | 测试正例在 G_t 中 |
|---|---|---|---|---|---|---|---|
| 2021 | 1,293 | 260 | 0 | 0 | 260 | 0 | 0 |
| 2022 | 1,576 | 290 | 0 | 0 | 290 | 0 | 0 |
| 2023 | 1,792 | 303 | 0 | 0 | 303 | 0 | 0 |
| 2024 | 1,762 | 339 | 0 | 0 | 339 | 0 | 0 |

**as-of 规则（补充核验，见第 8 节）：** 用仓库材料重建 `data_a2_apionly/` 后，所有 `graph_t.csv` 与提交版逐字节一致。代码中 G_t 只由 `strict = s[s.disc.le(C_t) & ...]` 构成（A4 不加文档边），其中 `disc` 是该行 rcept_no 对应的 `filings_A001.rcept_dt`。重建只能证明“提交的图由这条规则生成”；rcept_dt 本身是否等于真实披露日，沿用 CORRECTIONS (d) 的匹配结果，没有另外核对。

## 5. 从保存的测试预测独立重算 —— 通过

**输入：**
- `out_2_m3/test_scores_m3.csv`：34,239 行 = 339 个正例 × 101 个候选。
- `out_1c_a4/test_scores.csv`：M1、M1+scores、PA 的冻结分数。

两者按 (pair_id, other) 一一合并，没有缺行，标签完全一致。我用自己写的向量化实现重算，没有复用 `m3_eval.py`：组内 AUC、组内 MRR 和组内 Hits@10（都按期望秩），以及混合池 AP。bootstrap 按投资方（anchor）聚类重抽样，1000 次，`default_rng(20261103)`，所有模型共用同一组重抽样。

**结果：**
- 均值、AP 和区间端点都与 `test_metrics_m3.json` 和报告一致，最大绝对差 ≤ 1.1e-16。
- 逐正例的指标与 `within_test_m3.csv` 一致，最大差 ≤ 1e-16。

**全部新边（339 个正例，139 个投资方），差值 [95% 区间]：**

| 比较 | 组内 AUC | 组内 MRR | 组内 Hits@10 | 混合 AP |
|---|---|---|---|---|
| M3 − M1 | +.009 [−.021, +.048] | −.038 [−.085, +.038] | −.018 [−.095, +.052] | −.005 [−.026, +.014] |
| M3 − (M1+scores) | −.003 [−.036, +.035] | −.045 [−.086, +.023] | −.062 [−.133, +.012] | −.006 [−.022, +.011] |
| M3 − PA | **+.041 [+.002, +.082]** | +.010 [−.042, +.060] | +.068 [−.031, +.141] | +.011 [−.007, +.027] |

**分组（组内 AUC，按正例所在组）：**

| 分组 | n（投资方） | M3 − M1 | M3 − (M1+scores) | M3 − PA |
|---|---|---|---|---|
| 两端都有连接 | 239（79） | −.013 [−.059, +.031] | **−.043 [−.097, −.003]** | −.037 [−.113, +.007] |
| 一端孤立 | 84（69） | +.045 [−.034, +.123] | **+.080 [+.001, +.159]** | +.262 [+.180, +.344] |

**稳健性补充（新做的检查，只改 bootstrap 随机种子，数据和分数都不变）：** 下表是各个“区间不含 0”的结论在 bootstrap 种子 1–5 下的下限或上限。

| 结论 | 种子 20261103 | 种子 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 全部：M3 − PA 组内 AUC 下限 | +.002 | +.004 | **−.003** | .000 | +.001 | **−.002** |
| 两端有连接：M3 − (M1+scores) 上限 | −.003 | −.001 | −.004 | **+.002** | −.005 | −.000 |
| 一端孤立：M3 − (M1+scores) 下限 | +.001 | +.007 | **−.003** | +.006 | +.001 | **−.004** |

这三处“区间不含 0”都处在边界上，换一个 bootstrap 种子就可能翻转。第 8 节的重训进一步显示：两个分组的结论在重训后翻转为含 0，S2 则保持不含 0。它们只能理解为“方向上有迹象”，不能当作稳定的区间结论。

**其他：**
- 非金融且持股 ≥5% 子集（40 个正例，36 个投资方）：M3 − M1 组内 AUC +.163 [+.072, +.250]，M3 − PA +.280 [+.189, +.374]，与报告一致。正例少，而且这个子集结果是本阶段才看到的，报告标为探索性是恰当的。
- **按预注册主指标 AP 看：** M3 − PA 为 +.011 [−.007, +.027]，区间含 0，原文的 S2 不成立；M3 − M1 的 AP 区间也含 0。
- 报告的“全候选排序”Hits@10/50 和 MRR（§4）**没有为 M3 计算**。本项能核验的只有组内版本。

## 6. 报告列出的偏差和未验证项

报告第 4 节列了 **9 项**，与任务里说的“7 项”不一致。下面逐项审查全部 9 项。

| # | 内容 | 判断 | 是否可能影响结论 |
|---|---|---|---|
| 1 | 选配置改用验证组内 AUC | 已核实：两种准则选出同一个配置 | 否 |
| 2 | 表中 AP 是“5 个种子平均分数的 AP”（.040），§7 写的是“5 个种子 AP 的平均”（.036） | 已核实：各种子 AP 为 .0394/.0356/.0360/.0374/.0323，均值 .0361 | 否。按 §7 口径 M3 的 AP 更低，只会削弱 M3，M3 − M1 和 M3 − PA 的 AP 结论（区间含 0）不变 |
| 3 | M3 没有计算全候选排序 Hits@10/50 和 MRR | 属实 | 不影响已报告的数字，但 §4 规定的次要指标缺失，**报告不完整** |
| 4 | 分组 AP 用“正例自身的 group” | 与 **A1 修订冲突**：A1 写的是“组 AP 用自身组标签（含负例）落在该组的全部点对”，报告没有指出这一冲突。我按 A1 口径重算：两端有连接组 AP 为 M3 .047、M1 .065、M1+scores .060、PA .036；一端孤立组为 M3 .016、M1 .010、M1+scores .010、PA .005 | 低。排序与报告相同，组内 AUC 主指标不受影响；但分组 AP 的数字应改用 A1 口径，或注明偏离了 A1 |
| 5 | MLP 解码器的形式，以及 G_t 中合格集以外节点的处理，由执行者决定 | 代码与描述一致（对称解码器；非合格节点保留，特征做缺失填补） | 低。这些节点只参与消息传递，不作为候选 |
| 6 | CPU 非确定性 | **已核实存在**（第 8 节）：在另一台机器上重训，选出同一配置，整体指标差异 ≤ .005 | **会影响分组结论**：重训后，两个分组的“区间不含 0”都翻转为含 0；全部新边的 S2 则变得稍稳 |
| 7 | 用 `pip --break-system-packages` 安装 torch 2.15.0+cpu | 环境记录，无法验证 | 否 |
| 8 | 行业和市场用当前值（A3）、API 匹配漏边、幸存者偏差 | 与 Stage 1C 相同，沿用未验证状态 | 可能影响所有模型的绝对水平。行业和市场同时进入 M1 和 M3 的特征，对两者差值的影响方向**未验证** |
| 9 | “没有 push” | **已过时**：上述 commit 和两个标签现在都在 `origin` 上 | 否。属于事实描述错误，建议更正 |

**报告没有列出、但可能影响解读的事项：**
- 第 5 节的种子敏感性：3 处“区间不含 0”都在边界上。
- 按原预注册口径（AP，验证 AP 选出的 Jaccard），S2 会成立；按现行口径（组内 AUC，PA），S2 处在边界。两种口径结论不同，报告应当把两种都写出来。
- M3 网格与原始任务说明不一致（见 3a）。

## 7. 5 个种子和测试分数的平均 —— 通过

**证据：**
- `val_grid_runs.csv` 中 32 个配置都有 5 个种子（共 160 行）；选中配置的 5 个种子都完成了早停，best_epoch 在 35–47 之间。
- `test_scores_m3.csv` 有 `M3_s1`…`M3_s5` 五列，都没有缺失值。`M3` 列与五列均值的最大差为 1.1e-16，所以测试分数确实是 5 个种子预测的平均。
- 各种子单独的测试 AP 与报告所列的 .039 / .036 / .036 / .037 / .032 一致。

## 8. 用仓库材料从头重算（补充核验，2026-11-03）

**做法：**
- 在仓库**之外**的临时目录克隆 `c63217e`，加入 GitHub release `raw-data-20261102` 的 `interim.tar.gz`。原仓库没有任何改动。
- 建图：`python src/build_v2.py --names a2 --doc no --out rebuild_apionly`，约 7 秒，没有 API 调用。
- M3：新建虚拟环境，安装 torch 2.15.0+cpu、numpy 2.5.4、pandas 3.1.0、scikit-learn 1.9.1，在 16 核机器上运行 `m3_sage.py search`。然后把副本里的 `test_scores_m3.csv` 移开，运行 `m3_sage.py test`。
- **披露：** 这一步在副本里**重新给测试集打了分**，是核验者的复现，不参与任何选择或修改。

**数据重建：**
- 与提交的 `data_a2_apionly/` 相比，`pairs.csv`、`nodes.csv`、`eligible.csv` 和 `graph_2019…2025.csv` 等 15 个文件**逐字节相同**。
- `history_undirected.csv` 内容（作为集合）完全相同，只有行顺序不同，原因是 Python 集合的迭代顺序。
- 每年严格 as-of 的有向边数：2019 835、2020 944、2021 1,324、2022 1,628、2023 1,848、2024 1,828、2025 2,105。与 A2 修订（不含文档边）一致。

**M3 重训（验证）：**
- 160 次运行都完成了，150/160 次的 best_epoch 完全相同。与提交版相比，验证 AP 最大差 6.8e-4，验证组内 AUC 最大差 8.5e-3。
- 选中的配置**相同**，都是 `h128_l2_lr0.01_do0.3_mlp`，验证组内 AUC 为 0.6912（提交版 0.6926）；前三名的顺序也相同。

**M3 重训（测试）：**
- 5 个种子平均分数的 AP 为 .0401（提交版 .0397）；5 个种子 AP 的均值为 .037（提交版 .036）。
- 逐点分数差异不小（单个种子最大 0.14），但聚合后的指标很接近。

| 指标或比较（组内 AUC，bootstrap 种子 20261103） | 提交版 | 重训版 |
|---|---|---|
| 全部：M3 组内 AUC | .732 | .735 |
| 全部：M3 − M1 | +.009 [−.021, +.048] | +.012 [−.017, +.050] |
| 全部：M3 − (M1+scores) | −.003 [−.036, +.035] | −.000 [−.031, +.037] |
| 全部：**M3 − PA（S2）** | +.041 [**+.002**, +.082] | +.043 [**+.006**, +.084]；bootstrap 种子 1–5 的下限为 +.002 至 +.009，都大于 0 |
| 全部：M3 − PA（AP） | +.011 [−.007, +.027] | +.012 [−.006, +.027] |
| 两端有连接：M3 − (M1+scores) | −.043 [−.097, **−.003**] | −.038 [−.088, **+.001**]，区间含 0 |
| 一端孤立：M3 − (M1+scores) | +.080 [**+.001**, +.159] | +.078 [**−.003**, +.156]，区间含 0 |
| 非金融且持股 ≥5%：M3 − M1 | +.163 [+.072, +.250] | +.162 [+.073, +.247] |

**结论：**
- 数据管线可以从 GitHub 材料精确复现。
- M3 的训练有非确定性，但不改变配置选择，也不改变整体结论的方向。
- 报告里两个分组的“区间不含 0”，在重训后都**不成立**。这两条只能视为偶然结果，不应写成结论。
- S2 的组内 AUC 结论在重训后反而略稳：下限从 +.002 变为 +.006，对 bootstrap 种子不再敏感。但它仍然只是探索性结果，而且只在事后改定的口径（组内 AUC，比较对象 PA）下成立。按预注册的 AP 口径，S2 的区间含 0。

---

**结论：报告的主结论被支持。** 主结论是：所有结果都是探索性的；M3 相对 M1 和 M1+scores 没有可靠提升。数据管线可以从 GitHub 材料逐字节复现，所有数字都能从保存的预测精确复算，也没有发现 G_{t+1} 或测试正例泄漏进消息传递。但有两处需要更正：
- 两个分组里“区间不含 0”的说法（两端有连接组中 M3 劣于 M1+scores；一端孤立组中 M3 优于 M1+scores），在重训后都不成立，应删除或改为“方向性迹象”。
- S2 只在事后改定的组内 AUC 口径下勉强成立，按预注册的 AP 口径不成立，不能作为支持 M3 的证据。

---

## 附录：独立重算脚本（只读）

运行方式：`python -I verify_m3.py <linkpred 目录> <输出 json>`。本次在仓库外的临时目录运行，没有写入仓库。

```python
"""Independent verification of linkpred Stage 2 (M3). Read-only on the repo.
usage: python -I verify_m3.py <linkpred dir> <out json>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP

L = Path(sys.argv[1]); OUT = Path(sys.argv[2]); R = {}
D = L / 'data_a2_apionly'

# ---------- A. test score files ----------
T = pd.read_csv(L / 'out_1c_a4/test_scores.csv', dtype={'anchor': str, 'other': str})
M = pd.read_csv(L / 'out_2_m3/test_scores_m3.csv', dtype={'anchor': str, 'other': str})
seedcols = [f'M3_s{i}' for i in range(1, 6)]
R['m3_rows'] = len(M); R['m3_pos'] = int(M.label.sum())
R['m3_seedcols_present'] = all(c in M for c in seedcols)
R['m3_seedcols_nonnull'] = bool(M[seedcols].notna().all().all())
R['m3_equals_seed_mean_maxabs'] = float((M.M3 - M[seedcols].mean(1)).abs().max())
R['per_seed_test_AP'] = {c: float(AP(M.label, M[c])) for c in seedcols}
R['mean_of_per_seed_AP'] = float(np.mean(list(R['per_seed_test_AP'].values())))
R['AP_of_seed_mean'] = float(AP(M.label, M.M3))
X = T.merge(M[['pair_id', 'other', 'label', 'M3']].rename(columns={'label': 'label_m3'}), on=['pair_id', 'other'], how='outer', indicator=True)
R['merge_status'] = X._merge.value_counts().to_dict()
R['label_mismatch'] = int((X.label != X.label_m3).sum())
R['pairs_per_positive'] = X.groupby('pair_id').size().value_counts().to_dict()
X = X[X._merge == 'both'].drop(columns='_merge')

# ---------- B. within-candidate-set metrics, recomputed independently (vectorised) ----------
MODELS = ['M3', 'M1', 'M1+scores', 'PA', 'Jaccard', 'CN', 'AA']
pos = X[X.label == 1].set_index('pair_id')
neg = X[X.label == 0]
W = pos[['anchor', 'group', 'subset_nonfin_ge5']].copy()
for m in MODELS:
    x = neg.pair_id.map(pos[m])
    gt = (neg[m] > x).groupby(neg.pair_id).sum(); lt = (neg[m] < x).groupby(neg.pair_id).sum()
    eq = (neg[m] == x).groupby(neg.pair_id).sum(); n = neg.groupby('pair_id').size()
    rank = gt + 1 + eq / 2
    W[f'{m}|AUC'] = (lt + eq / 2) / n; W[f'{m}|MRR'] = 1 / rank; W[f'{m}|H10'] = (rank <= 10).astype(float)
W = W.reset_index()

# compare with the repo's own per-positive file
Wr = pd.read_csv(L / 'out_2_m3/within_test_m3.csv', dtype={'anchor': str}).set_index('pair_id')
Wc = W.set_index('pair_id').loc[Wr.index]
R['within_file_maxabs_diff'] = {c: float((Wc[c] - Wr[c]).abs().max()) for c in Wr.columns if '|' in c}

cells = {'all': np.ones(len(W), bool), 'nonfin_ge5': W.subset_nonfin_ge5.astype(bool).values,
         'both_connected': (W.group == 'both_connected').values, 'one_isolated': (W.group == 'one_isolated').values,
         'both_isolated': (W.group == 'both_isolated').values}
COMPS = [('M3', 'M1'), ('M3', 'M1+scores'), ('M3', 'PA'), ('M1+scores', 'M1'), ('M3', 'Jaccard')]

def boot(U, TP, seed, B=1000):
    ua = np.unique(U.anchor.values)
    ixW = {a: np.where(U.anchor.values == a)[0] for a in ua}; ixT = {a: np.where(TP.anchor.values == a)[0] for a in ua}
    rng = np.random.default_rng(seed); out = {f'{a}-{b}|{k}': [] for a, b in COMPS for k in ('AUC', 'MRR', 'H10', 'AP')}
    Uv = {c: U[c].values for c in U.columns if '|' in c}; TPl = TP.label.values; TPs = {m: TP[m].values for m in MODELS}
    for _ in range(B):
        d = rng.choice(ua, len(ua)); iw = np.concatenate([ixW[a] for a in d]); it = np.concatenate([ixT[a] for a in d])
        ok = len(np.unique(TPl[it])) > 1
        ap = {m: AP(TPl[it], TPs[m][it]) if ok else np.nan for m in ('M3', 'M1', 'M1+scores', 'PA', 'Jaccard')}
        for a, b in COMPS:
            for k in ('AUC', 'MRR', 'H10'): out[f'{a}-{b}|{k}'].append(Uv[f'{a}|{k}'][iw].mean() - Uv[f'{b}|{k}'][iw].mean())
            out[f'{a}-{b}|AP'].append(ap[a] - ap[b])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in out.items()}

R['cells'] = {}
for c, msk in cells.items():
    U = W[msk].reset_index(drop=True)
    TP = X[X.pair_id.isin(U.pair_id)].reset_index(drop=True)          # report's rule: positive's group
    TPg = X[X.group == c] if c in ('both_connected', 'one_isolated', 'both_isolated') else None  # A1 rule: pair's own group
    e = dict(n_pos=len(U), n_anchor=int(U.anchor.nunique()))
    e['mean'] = {f'{m}|{k}': float(U[f'{m}|{k}'].mean()) for m in MODELS for k in ('AUC', 'MRR', 'H10')}
    e['AP_positive_group'] = {m: float(AP(TP.label, TP[m])) for m in MODELS}
    if TPg is not None and TPg.label.nunique() > 1: e['AP_A1_pair_group'] = {m: float(AP(TPg.label, TPg[m])) for m in MODELS}
    e['point'] = {f'{a}-{b}|{k}': e['mean'][f'{a}|{k}'] - e['mean'][f'{b}|{k}'] for a, b in COMPS for k in ('AUC', 'MRR', 'H10')}
    e['point'].update({f'{a}-{b}|AP': e['AP_positive_group'][a] - e['AP_positive_group'][b] for a, b in COMPS})
    if len(U) >= 30:
        e['ci_seed20261103'] = boot(U, TP, 20261103)
        if c in ('all', 'both_connected', 'one_isolated'):
            e['ci_other_seeds_M3-PA_AUC'] = {s: boot(U, TP, s, 1000)['M3-PA|AUC'] for s in (1, 2, 3, 4, 5)}
            e['ci_other_seeds_M3-M1+scores_AUC'] = {s: boot(U, TP, s, 1000)['M3-M1+scores|AUC'] for s in (1, 2, 3, 4, 5)}
    R['cells'][c] = e

# ---------- C. repo's reported numbers ----------
rep = json.load(open(L / 'out_2_m3/test_metrics_m3.json'))
R['repo_vs_mine_maxabs'] = {}
for c, e in rep.items():
    mine = R['cells'][c]; d = []
    for k, v in e['mean'].items():
        mod, met = k.split('|'); d.append(abs(v - (mine['AP_positive_group'][mod] if met == 'AP' else mine['mean'][k])))
    if 'ci_seed20261103' in mine:
        for k, v in e['diff'].items():
            if k in mine['ci_seed20261103']: d += [abs(v['lo'] - mine['ci_seed20261103'][k][0]), abs(v['hi'] - mine['ci_seed20261103'][k][1])]
    R['repo_vs_mine_maxabs'][c] = float(max(d))

# ---------- D. message passing / leakage on saved graphs ----------
P = pd.read_csv(D / 'pairs.csv', dtype={'anchor': str, 'other': str})
R['split_years'] = P.groupby('split').fy.unique().apply(lambda a: sorted(map(int, a))).to_dict()
H = pd.read_csv(D / 'history_undirected.csv', dtype=str); R['history_cols'] = list(H.columns)
und = lambda a, b: set(map(frozenset, zip(a, b)))
G = {t: pd.read_csv(D / f'graph_{t}.csv', dtype=str) for t in range(2019, 2026)}
Ge = {t: und(g.src, g.dst) for t, g in G.items()}
leak = {}
for t in range(2021, 2025):
    pp = P[P.fy == t]; pos_t = und(pp[pp.label == 1].anchor, pp[pp.label == 1].other); all_t = und(pp.anchor, pp.other)
    nxt = Ge[t + 1] - set().union(*[Ge[s] for s in range(2019, t + 1)])
    leak[t] = dict(n_edges_Gt=len(Ge[t]), pos=len(pos_t), pos_in_Gt=len(pos_t & Ge[t]), anypair_in_Gt=len(all_t & Ge[t]),
                   pos_in_Gt1=len(pos_t & Ge[t + 1]), new_Gt1_edges_in_Gt=len(nxt & Ge[t]),
                   test_pos_in_Gt=len(und(P[(P.split == 'test') & (P.label == 1)].anchor, P[(P.split == 'test') & (P.label == 1)].other) & Ge[t]))
R['leak'] = leak
# validation summary: recompute selection
V = pd.read_csv(L / 'out_2_m3/val_grid_runs.csv')
R['val_runs'] = len(V); R['val_configs'] = int(V.config.nunique()); R['seeds_per_config'] = V.groupby('config').seed.nunique().value_counts().to_dict()
S = V.groupby('config')[['val_within_AUC', 'val_AP']].mean()
R['val_argmax_withinAUC'] = S.val_within_AUC.idxmax(); R['val_argmax_AP'] = S.val_AP.idxmax()
R['val_top3_withinAUC'] = S.val_within_AUC.sort_values(ascending=False).head(3).to_dict()
R['val_runs_columns'] = list(V.columns)
# A6: recompute validation within AUC for the four heuristics from saved validation scores
VS = pd.read_csv(L / 'out_1c_a4/val_scores.csv', dtype={'anchor': str, 'other': str})
vp = VS[VS.label == 1].set_index('pair_id'); vn = VS[VS.label == 0]; a6 = {}
for m in ('CN', 'AA', 'Jaccard', 'PA'):
    x = vn.pair_id.map(vp[m]); n = vn.groupby('pair_id').size()
    a6[m] = float((((vn[m] < x).groupby(vn.pair_id).sum() + (vn[m] == x).groupby(vn.pair_id).sum() / 2) / n).mean())
R['A6_val_withinAUC_recomputed'] = a6; R['A6_val_n_pos'] = int(VS.label.sum()); R['A6_val_splits'] = VS.split.unique().tolist()
R['A6_val_AP'] = {m: float(AP(VS.label, VS[m])) for m in ('CN', 'AA', 'Jaccard', 'PA')}
json.dump(R, open(OUT, 'w'), indent=1, default=str)
print('ok')
```
