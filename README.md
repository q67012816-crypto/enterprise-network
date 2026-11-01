# enterprise-network：韩国上市公司持股/出资网络（描述性研究）

**只做描述性统计，不训练任何模型。** 阶段 A（当前）：接口调研 + 脚手架；阶段 B：需 API 密钥后拉数、建图、统计。

## 数据来源
- OpenDART（금융감독원 전자공시 오픈API，https://opendart.fss.or.kr）。接口详情见 `docs/api-notes.md`。
- 主边来源：`otrCprInvstmntSttus.json`（타법인 출자현황），`reprt_code=11011`（사업보고서），`bsns_year=2025`（主；已实测发布：2,609/2,655 家 status 000，46 家 013）+ `2024`（稳健性）。
- 节点属性：`corpCode.xml`（corp_code、stock_code）+ `company.json`（corp_cls 市场、induty_code 行业）。

## 范围与假设
- 节点 = `company.json` 中 corp_cls ∈ {Y→KOSPI, K→KOSDAQ} 的公司（快照日期即拉取日；历史年度的上市状态不回溯 = 已知偏差）。
- 第一版只保留两端都在上市范围内的边；被丢弃的出资对象（非上市、外国、个人、无法匹配）计数并输出覆盖率。

## 边的定义
- 有向边 `source → target`：source = 投资方（披露 타법인 출자현황 的上市公司），target = 被投资方（inv_prm 解析得到的上市公司）。
- 权重 `weight` = `trmend_blce_qota_rt`（期末持股比例，%）；缺失/非数值时置空并在质量清单记录，不填 0。
- 字段：source_corp_code, target_corp_code, ownership_pct, source_api, bsns_year, reprt_code, rcept_no, match_rule。
- 同一 (source,target,year) 多行（如多次取得/多股种）：保留 ownership_pct 最大一行，并记录行数 `n_rows`。
- **无向版本**（用于度数）：按无序对 {u,v} 合并；A→B 与 B→A 合并为一条边，`weight_max` = 两方向最大比例，`directions` ∈ {`uv`,`vu`,`both`}；自环（投资自身，如自己股票）删除并计数。度 = 无向简单图中的邻居数。

## 实体解析规则（src/pipeline.py: step_resolve）
1. R1 精确匹配 corp_name；2. R2 去掉 (주)/㈜/주식회사/Co.,Ltd/空格等后的规范名在上市公司中唯一；3. R3 规范化 stock_name 唯一；
其余标记 `ambiguous`/`unmatched`，全部写入 `data/processed/unmatched_investees.csv`，不静默丢弃。不使用模糊匹配（避免假阳性）。

## OpenDART 使用条款要点（https://opendart.fss.or.kr/intro/terms.do，2020-01-21 施行）
- 第10条④⑤⑥：调用次数有允许量上限（主页公布，可变更）；过度访问可被限制。开发指南状态码 020：一般 20,000 次以上触发（是否按日 = 未验证）。
- 第11条：服务原则上免费（「원칙적으로 무료」），将来部分服务可能收费并事先通知。目前未发现收费接口。
- 第16条：OpenAPI 服务及相关程序著作权归金融监督院；未规定事项依著作权法与公共数据法。
- 第19条②⑤：密钥不得给第三方使用；每人仅一个密钥。→ 密钥只从环境变量 `OPENDART_API_KEY` 读取，不写入代码/日志/缓存/git。
- 第23条：公示信息由提交人负责，金融监督院不保证准确性与完整性。
- 条款未禁止学术研究用途；也未明文规定缓存/再分发/署名。再分发原始数据前建议注明来源"OpenDART(금융감독원)"——再分发是否受限 = 未验证（依公共数据法）。

## 复现
```bash
cd enterprise-network
python3 -m venv .venv && .venv/bin/pip install pytest
.venv/bin/python -m pytest -q tests          # 离线测试（mock，无需密钥）
export OPENDART_API_KEY=...                  # 只在 shell 中设置，勿写入文件
cd src && python pipeline.py all --year 2025 --prev-year 2024   # 可中断；重跑自动续跑（缓存命中不计次数）
```
目录：`data/raw/`（原始响应缓存，不提交）、`data/state/`（每日调用计数）、`data/interim/`、`data/processed/`、`reports/`。

## Phase B results and commands (2026-11-01)
- Reports: `reports/DEGREE_REPORT.md`, `reports/LABEL_FEASIBILITY.md`, plots in `reports/*.png`, statistics in `reports/stats_*.json`, `reports/labels_2025.json`
- Data: `data/processed/nodes.csv`, `edges.csv` (directed), `edges_undirected.csv`, `groups.csv` (approximate group proxy, 未验证), `discarded_investees.csv`, `unmatched_investees.csv`, `labels_*.csv`, `build_counts.json`
- Commands: see "How to reproduce" in DEGREE_REPORT. Concurrency is set by `DART_WORKERS` (default 10), and the global spacing between requests by `DART_MIN_INTERVAL` (default 0.25 s).
- Discard categories: summary_or_empty / non_listed_domestic / foreign_likely (regex) / ambiguous / unmatched; self-loops are removed. There is no "individual" category, because this endpoint only lists investee corporations.
- Group proxy: hyslrSttus "본인" (or the largest shareholder) → climb up through listed companies, ≤6 levels. This is not the 공정위 기업집단 designation.
- Known engineering issue: `data/state/calls.json` is overwritten when several processes run at once, so the count is undercounted. The actual number of requests today is about 13.8k (counted from the `.meta.json` files in the raw cache), below the budget. Run only one fetching process at a time.

## Phase 2 (in progress)
Progress log: `reports/PHASE2_PROGRESS.md`. Corrections: `reports/CORRECTIONS.md`. New scripts: `src/corrections.py`, `edge_dates.py`, `universe.py`, `filings.py`, `financials.py`.
