# OpenDART 接口调研记录 (api-notes)

调研时间：2026-11-01（KST）。来源：官方开发指南 `https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=<grp>&apiId=<id>`，
页面正文快照保存在 `docs/guide_snapshots/<apiId>.txt`（公开页面，无密钥）。**本阶段没有 API 密钥，所有接口均未实际调用**，
返回字段以官方文档为准；Phase B 实测 company / otrCprInvstmntSttus / hyslrSttus / accnutAdtorNmNdAdtOpinion / fnlttMultiAcnt / list 字段与指南一致（已验证）。数值为带逗号字符串，空值为 "-"。

## 通用约定（官方文档原文确认）
- 认证：所有接口必填 `crtfc_key`（STRING(40)，发급받은 인증키）。
- 格式：多数接口同时提供 `.json` / `.xml`（UTF-8）；`corpCode.xml`、`document.xml` 返回 Zip 二进制。
- `bsns_year`：사업연도 4 位；定期报告类接口注明「2015년 이후 부터 정보제공」（fnlttSinglIndx 为「2023년 3분기 이후」）。
- `reprt_code`：`11013`=1분기보고서，`11012`=반기보고서，`11014`=3분기보고서，`11011`=사업보고서（年度报告，本项目使用）。
- 状态码：000 正常；010 未注册密钥；011 停用密钥；012 IP 不可访问；013 无数据；014 文件不存在；
  **020 超过请求限制：「일반적으로는 20,000건 이상의 요청에 대하여 이 에러 메시지가 발생되나, 요청 제한이 다르게 설정된 경우에는 이에 준하여 발생됩니다」**；
  021 可查询公司数超限（最多 100 家，用于多公司接口）；100 字段值不当；101 不当访问；800 系统维护；900 未定义错误；901 账号个人信息保有期满。

### 调用上限
- 官方文档唯一明确数字：状态码 020 说明中的「一般为 20,000 件以上的请求」。**文档未写明这是"每日"上限**；用户记忆中的"约 1 万次/日"在官方页面中**未找到**。
- 이용약관 第10条④：「오픈API 서비스는 이용횟수에 허용량 제한이 있으며, 이를 홈페이지에 게시합니다」；⑥ 过度网络访问可被限制使用。
- 计数周期（每日/每 KST 自然日）= **未验证**。实现上：默认每日预算 18,000（`DART_DAILY_BUDGET` 可调），请求间隔 0.7 s，遇 020 立即停止，次日续跑。

## 接口表

| # | 用途 | 接口名（韩） | URL | 必填参数 | 关键返回字段 | 状态 |
|---|---|---|---|---|---|---|
|1|企业代码表|고유번호 (DS001/2019018)|`/api/corpCode.xml`（Zip）|crtfc_key|corp_code(8), corp_name, corp_eng_name, stock_code(6, 上市才有), modify_date|文档确认|
|1b|企业概况：市场、行业|기업개황 (DS001/2019002)|`/api/company.json`|crtfc_key, corp_code|corp_name, stock_name, stock_code, **corp_cls（Y=유가/KOSPI, K=코스닥, N=코넥스, E=기타）**, jurir_no, bizr_no, **induty_code**, est_dt, acc_mt, adres…|文档确认|
|2|对其他法人出资|타법인 출자현황 (DS002/2019015)|`/api/otrCprInvstmntSttus.json`|crtfc_key, corp_code, bsns_year, reprt_code|rcept_no, corp_cls, corp_code, corp_name, **inv_prm（法人名 = 被投资方名称，仅名称，无 corp_code）**, frst_acqs_de, invstmnt_purps, frst_acqs_amount, bsis_blce_qy, bsis_blce_qota_rt, bsis_blce_acntbk_amount, incrs_dcrs_acqs_dsps_qy/amount, incrs_dcrs_evl_lstmn, trmend_blce_qy, **trmend_blce_qota_rt（期末持股比例）**, trmend_blce_acntbk_amount, recent_bsns_year_fnnr_sttus_tot_assets, recent_bsns_year_fnnr_sttus_thstrm_ntpf, stlm_dt|文档确认|
|3a|最大股东及特殊关系人|최대주주 현황 (DS002/2019007)|`/api/hyslrSttus.json`|同上|rcept_no, corp_cls, corp_code, corp_name, **nm（姓名/名称）**, **relate（关系：본인, 친인척 등）**, stock_knd, bsis_posesn_stock_co, bsis_posesn_stock_qota_rt, trmend_posesn_stock_co, **trmend_posesn_stock_qota_rt**, rm, stlm_dt|文档确认|
|3b|最大股东变动|최대주주 변동현황 (DS002/2019008)|`/api/hyslrChgSttus.json`|同上|change_on, mxmm_shrholdr_nm, posesn_stock_co, qota_rt, change_cause, rm, stlm_dt|文档确认|
|4a|单一公司主要账户|단일회사 주요계정 (DS003/2019016)|`/api/fnlttSinglAcnt.json`|crtfc_key, corp_code, bsns_year, reprt_code|rcept_no, bsns_year, stock_code, reprt_code, account_nm, fs_div(OFS/CFS), fs_nm, sj_div(BS/IS), sj_nm, thstrm_*/frmtrm_*/bfefrmtrm_* 金额, ord, currency|文档确认|
|4b|多公司主要账户|다중회사 주요계정 (DS003/2019017)|`/api/fnlttMultiAcnt.json`|同上；corp_code 可逗号分隔多家（示例 `00334624,00126380`；状态 021 表明上限 100 家）|同 4a|文档确认|
|4c|单一公司全部财务报表|단일회사 전체 재무제표 (DS003/2019020)|`/api/fnlttSinglAcntAll.json`|页面存在；参数未逐项抄录（含 fs_div）= 未验证|—|仅确认存在|
|4d|单一公司主要财务指标|단일회사 주요 재무지표 (DS003/2022001)|`/api/fnlttSinglIndx.json`|上述 + **idx_cl_code**（M210000 收益性/M220000 安定性/M230000 成长性/M240000 活动性）；2023Q3 以后|idx_code, idx_nm, idx_val…|文档确认|
|5|审计意见|회계감사인의 명칭 및 감사의견 (DS002/2020009)|`/api/accnutAdtorNmNdAdtOpinion.json`|crtfc_key, corp_code, bsns_year（文档写 STRING(1)，说明为 4 位，文档笔误）, reprt_code|rcept_no, corp_cls, corp_code, corp_name, bsns_year（당기/전기/전전기）, adtor, **adt_opinion**, adt_reprt_spcmnt_matter（至 2019-12-08）, emphs_matter, core_adt_matter（2019-12-09 起）, stlm_dt|文档确认|
|5b|审计服务合同|감사용역체결현황 (DS002/2020010)|`/api/adtServcCnclsSttus.json`|定期报告类通用参数（推断，未逐项验证）|—|仅确认存在|

另：DS002 下还存在 `hmvAuditIndvdlBySttusV2`、`indvdlByPayV2`（2026 年新增 Ver 2.0，与本项目无关）。

## 标签候选来源（Phase B 详查，这里仅列接口是否存在）
| 候选标签 | 来源 | 状态 |
|---|---|---|
|审计意见非适正（한정/부적정/의견거절）|`accnutAdtorNmNdAdtOpinion.json`.adt_opinion|已验证：取值为 "적정의견"/"한정의견"/"의견거절"/空；period 形如 "제14기\n(당기)"；一家每期可多行|
|부도발생|주요사항보고서 `dfOcr.json` (DS005/2020019)，参数 corp_code, bgn_de, end_de（2015 年以后）|接口存在|
|영업정지|`bsnSp.json` (DS005/2020020)|接口存在|
|회생절차 개시신청|`ctrcvsBgrq.json` (DS005/2020021)|接口存在|
|해산사유 발생|`dsRsOcr.json` (DS005/2020022)|接口存在|
|관리종목 지정 / 상장폐지 / 거래정지|无专门结构化接口（在指南中未找到）。可用 `list.json` 公示检索：pblntf_ty=I（거래소공시）、I003 시장조치/안내，按 report_nm 关键词筛选；corp_code 不填时检索期间限 3 个月|**无专用接口；只能文本匹配，口径 未验证**|
|资本侵蚀（자본잠식）|由 fnlttSinglAcnt 的 자본총계/자본금 计算（衍生标签，非官方标记）|可计算；口径需定义|
|退市推断|corpCode.xml 中 stock_code 消失 / company.json corp_cls 变化（只反映当前快照）|未验证|

`list.json`（DS001/2019001）参数：corp_code, bgn_de, end_de, last_reprt_at, pblntf_ty(A–J), pblntf_detail_ty（A001=사업보고서…）, corp_cls, sort, sort_mth, page_no, page_count(≤100)；返回 corp_cls, corp_name, corp_code, stock_code, report_nm, rcept_no, flr_nm, rcept_dt, rm。

## 最新完整年度
- 12 月结算公司的 사업보고서 法定提交期限为结算后 90 日（一般 3 月末）——此为常识，非本次从 OpenDART 页面核实 = 未验证。
- 今天 2026-11-01，**推定最新完整年度 bsns_year=2025（reprt_code=11011），备份 2024**。**已验证已发布**：otrCprInvstmntSttus 2025 年 2,609/2,655 家 status 000（46 家 013 无数据）；2024 年 2,519 家 000、136 家 013。
- 非 12 月结算公司（acc_mt≠12）的 bsns_year 语义 = 未验证，Phase B 用 company.acc_mt 标记。

## 已知局限（来自文档本身）
- 타법인 출자현황只给被投资方**名称**（inv_prm），无 corp_code/法人登记号 → 必须做名称实体解析。
- 该表只覆盖"上市公司作为投资方"披露的对外出资；反向（谁持有我）需靠 hyslrSttus（仅最大股东及特殊关系人，含个人）。

## Phase B measurements (2026-11-01, with key)
- Status codes seen: 000 and 013 (no data). 020, 800 and 900 were never seen. Today about 13.8k requests (counted from cache files) never triggered 020, so whether the 020 limit is counted per day is still 未验证.
- Latency is about 1.4–2.6 s per request. With 10 threads and a global spacing of 0.25 s, throughput was about 3–4 requests/s and no throttling occurred.
- The hyslrSttus fields nm / relate / trmend_posesn_stock_qota_rt were verified.
- list.json: B001 and I003 were fetched by quarter for 2024–2025 (87 cached calls in total).
- Guide snapshots for 2019002/2019006/2019007/2019016/2019017/2019018/2020009/2020019–22/2022001 were added to docs/guide_snapshots/ (HTML converted to text).
