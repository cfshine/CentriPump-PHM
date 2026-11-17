# 离心泵诊断报告

- 诊断编号：`demo-bearing-c70ddab28838`
- 设备：`PUMP-DEMO-BEARING-01`
- 数据窗口：2026-11-17T00:00:00 ～ 2026-11-17T00:30:00（案例时间为 UTC）
- 报告生成时间：2026-11-17T11:18:56（运行环境本地时间）
- 运行模式：`offline`

## 结论

安全裁决：**BLOCK**；风险等级：**CRITICAL**。
故障模式：`BEARING_DEGRADATION`。

故障模式是基于现有证据的候选判断，润滑失效、疲劳损伤等具体根因仍须停机检查确认。

## 输入与执行情况

工程师描述：合成演示：驱动端轴承逐渐发热并伴随振动升高，请排查。未提供照片。

| 环节 | 结果 |
|---|---|
| 工程师文本 | RECORDED |
| 时序分析 | OK，361 点； |
| 视觉分析 | NO_IMAGE；无图片：未提供现场图片，跳过视觉分析 |
| 规程检索 | HIT，1 条 |
| 综合归因 | 综合【DATA, MANUAL, ENGINEER_TEXT】得出初步判断：驱动端轴承干摩擦或疲劳劣化。 |

本次为合成案例的离线集成验证。时序统计、规则判定和安全门禁真实执行；数据描述采用确定性模板，归因采用现有规则。未调用语言模型、视觉模型、MySQL 或向量检索。未提供现场照片，不构造视觉证据。演示规程是自制测试资料，不能替代正式设备手册或经审核的生产规程。

## 窗口统计

| 指标 | 数值 | 单位 |
|---|---:|---|
| total_points | 361 | count |
| phase_count | 2 | count |
| max_temp_de | 85.5 | ℃ |
| max_vib_de | 5.25 | mm/s |
| max_temp_nde | 45 | ℃ |
| max_vib_nde | 1.4 | mm/s |
| avg_flow | 100.015 | m³/h |
| min_press_in | 0.02 | MPa |

## 分段与规则

| 阶段 | 开始 | 结束 | 点数 | 命中规则 |
|---|---|---|---:|---|
| NORMAL | 2026-11-17T00:00:00 | 2026-11-17T00:04:55 | 60 |  |
| DEGRADING | 2026-11-17T00:05:00 | 2026-11-17T00:30:00 | 301 | BEARING_TEMP_DE_TRIP, BEARING_TEMP_DE_RAMP_SHARP, VIB_DE_TRIP, VIB_DE_DOMINANT |

安全门禁判定理由：

- 触发紧急停机线/急剧恶化判据【BEARING_TEMP_DE_RAMP_SHARP】: 驱动端温度急剧恶化 (依据: 规范 5.2)。
- 触发紧急停机线/急剧恶化判据【BEARING_TEMP_DE_TRIP】: 驱动端温度超停机线 (依据: GB 50275-2010 / 规范第三节)。
- 触发紧急停机线/急剧恶化判据【VIB_DE_TRIP】: 驱动端振动超国标停机线 (依据: GB 50275 附录 A)。

## 规程证据

### DEMO-BEARING-MANUAL / bearing-01

位置：1. 驱动端轴承温振异常（演示资料）；检索方式：KEYWORD；匹配词覆盖率：0.60。

演示案例采用项目 rules/thresholds.py 的判据：驱动端轴承温度达到80℃或振动达到4.5mm/s触发停机线。温振同步恶化时应优先核验轴承及润滑系统，具体根因须检查确认。

## 安全处置与后续核验

1. [HIGH] 【安全门禁最高级指令】立即执行紧急切机停运操作！禁止带病维持运转，防范轴承烧死或介质爆炸事故！（依据：`CRITICAL_BLOCK`）
2. [HIGH] 切断机泵主电源并实施安全挂牌上锁 (LOTO)，通知工艺管网值班长启动备用机泵切换。（依据：`LOTO_PROCEDURE`）

上述为诊断建议；本次运行未连接控制系统，也未实际停机或下发工单。

## 溯源与完整性

- 规则版本：`1.0.0`
- 门禁输入 SHA-256：`9c9742a3bfa293181f82bedf5d077d63beb69602ca3f347d9d49b0078f5834ef`
- 裁决与证据链 SHA-256：`cfb4b50cb58e9632a5c19db3200cf7a4cff8b509296f3b45b99756ee570059c8`（不代表整份 Markdown 文件哈希）
- 逻辑交付地址：`dispatch://emergency_shutdown/PUMP-DEMO-BEARING-01/demo-bearing-c70ddab28838?sha256=cfb4b50cb58e9632a5c19db3200cf7a4cff8b509296f3b45b99756ee570059c8`
- 完整状态档案：`demo_diagnosis_report.json`

- 输入文件：`E:\vscode\CentriPump-PHM\data\sample_inputs\demo_bearing.csv`；SHA-256：`558e906750652c6a3be4f8760566ff30bda8a2293b23ba7d6919ea7312513f5d`
- 输入文件：`E:\vscode\CentriPump-PHM\data\manuals\demo_bearing_manual.json`；SHA-256：`639b61807a95903ab517a44fdc4eaa3eedbf16df0de0d603a2aa0106c94d9b1b`

- THRESHOLD: `BEARING_TEMP_DE_TRIP` — data.threshold_flags @ 2026-11-17T00:00:00~2026-11-17T00:30:00
- THRESHOLD: `BEARING_TEMP_DE_RAMP_SHARP` — data.threshold_flags @ 2026-11-17T00:00:00~2026-11-17T00:30:00
- THRESHOLD: `VIB_DE_TRIP` — data.threshold_flags @ 2026-11-17T00:00:00~2026-11-17T00:30:00
- THRESHOLD: `VIB_DE_DOMINANT` — data.threshold_flags @ 2026-11-17T00:00:00~2026-11-17T00:30:00
- MANUAL: `DEMO-BEARING-MANUAL#bearing-01` — page 0 · 1. 驱动端轴承温振异常（演示资料）
- RULE: `BEARING_TEMP_DE_RAMP_SHARP` — rules/safety_guard.py
- RULE: `BEARING_TEMP_DE_TRIP` — rules/safety_guard.py
- RULE: `VIB_DE_TRIP` — rules/safety_guard.py
