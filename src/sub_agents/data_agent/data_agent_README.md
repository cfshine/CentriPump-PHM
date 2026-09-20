# Step 2 · data_agent 实现说明

> **这份文档回答三件事**：
> 1. 数据从 Step 1 进来、到最终出去，**中间变成了什么形状**（用真实测试用例逐步展示）
> 2. 每个关键设计**为什么这么做**
> 3. **怎么提测**
>
> ★ **v2（2026-09-17）**：整个子图做了一次"去繁从简" —— 3 节点压成 2 节点、
> 公共字段 8 → 3、私有字段 2 → 0、数据库查询 2 次 → 1 次。
> 变更清单与理由见 §3.14~§3.16。

---

## 0. 这个模块是什么

**一句话**：把 Step 1 递来的「一台设备 + 一个时间窗口」，变成一份结构化的工况分析结论，交给下游四个 Step。

**职责边界（三件事，不多不少）**：

```
   ① 取 SCADA 数据        ② 确定性计算（描述状态，不做判断/推理）
   ③ 让大模型把算好的数字翻译成人话
```

```
                  ┌──────────────────────────────────────────┐
   Step 1 ───────►│  data_agent（Step 2）                     │
   state.context   │  analyze（取数 + 确定性计算）              │
   ├ device_id     │      ↓                                    │
   ├ start_time    │  summarize（大模型语义化）                 │
   └ end_time      └────────────────┬─────────────────────────┘
                                    │ 只回写 data 一个盒子
                                    ▼
                          state.data（DataState）
        ┌───────────────┬───────────────┬──────────────┬──────────────┐
        ▼               ▼               ▼              ▼              ▼
     metrics     threshold_flags      alarms      descriptions   quality
   （→ Step 5）   （→ Step 6 门禁）  （报警事实）  （→ Step 4 RAG） （数据质量）
```

> ★ 2026-09-20：组长的 ``DiagnosisState`` 改成 9 个嵌套盒子
> （context / data / vision / manual / reasoning / safety / human / delivery / workflow），
> **Step 2 只读 `context`、只写 `data`**。挂载走路线 B：主图 `data_node` 转调
> `data_graph.run_data_agent(state)`，后者只回写 `{"data": ...}`
> （data 与 vision 是并行分支，多写一个顶层键就会覆盖 vision 的产出）。

### 文件清单

| 文件 | 行数 | 职责 | 依赖 LangGraph？ |
| :--- | ---: | :--- | :---: |
| `data_graph.py` | 67 | 子图定义（2 个节点 + 边）+ 路线 B 入口 `run_data_agent` | ✅ 唯一 |
| `data_nodes.py` | 326 | 2 个节点适配函数 + LLM 输出契约 + `_data_update` 回写助手 | ✅ |
| `data_tools.py` | 60 | 取数：**一条 SQL** 按设备 + 窗口取帧 | ❌ |
| `repository.py` | 107 | `scada_telemetry` 表结构与 ORM | ❌ |
| `analyzer.py` | 248 | **段内**统计指标 + 规则判定（产出机器码，纯 pandas，核心算法） | ❌ |
| `timeseries_tools.py` | 222 | 滑窗 / 回归 / CV / 报警码提取 / 文本格式化 / 规则码渲染 | ❌ |

> **`analyzer.py` 与 `timeseries_tools.py` 不含任何 LangGraph 与数据库依赖**，
> 只吃 DataFrame、只吐 dict/数值 —— 所以它们可以脱离图和数据库单独测试。
> 这是刻意的：算法归算法，编排归编排。
>
> **`data_state.py` 已删除**：Step 2 现在没有私有字段，子图直接拿公共契约
> `DiagnosisState` 当自己的 state（理由见 §3.15）。

---

## 1. 怎么提测

### 1.1 环境准备

```bash
cd /home/yali_ai/work_file/LLM_Project/CentriPump-PHM

# 解释器：项目自身 venv（含 langgraph / langchain / pandas / sqlalchemy / pymysql）
PY=/home/yali_ai/work_file/.venv/bin/python

# 大模型凭据：从系统环境变量读，代码里不做任何显式声明
eval "$(grep -E '^export DEEPSEEK_API_KEY=' ~/.bashrc)"

# 数据库：MySQL 127.0.0.1:3306 / scada_db（配置在项目根 .env）
```

### 1.2 两条提测命令

```bash
# ① 契约与节点行为测试 —— 不需要数据库、不需要网络
$PY -m pytest tests/test_nodes.py -q
#    期望：27 passed（有 Key）；涉及建图/取数的用例在无 Key 时自动跳过
#    ⚠ 全量 `pytest tests/` 目前有 2 个 FAILED，都在 tests/test_vision_agent.py ——
#      那是 Step 3 还没适配 9 盒子契约（本轮范围外），Step 2 自身全绿
#    覆盖：公共契约 / 子图 2 节点 / 报警码只认窗口数据 / 软降级 / 阈值唯一出处 / 主图挂载

# ② 端到端回归 —— 需要 MySQL + 大模型 Key
$PY -m tests.test_pipeline
#    期望：通过: 30/30
#    覆盖：30 个真实窗口，含故障周期、窗口边界、停机、重启、碎片、降级
```

> ①里的取数用例用 `monkeypatch` 注入合成帧，所以**离线可跑** ——
> "未提供窗口不查库"、"窗口无数据降级"、"报警码不读用户入参" 都不需要 MySQL。

### 1.3 命令失败自查

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| `ImportError: cannot import name 'model'` | `src/utils/llm_client.py` 不完整 | 确认文件末尾有 `model = init_chat_model(...)` |
| 全部用例 `points=0` | 数据库没起 / 窗口无数据 | `mysql -h127.0.0.1 -uroot -e "USE scada_db; SELECT COUNT(*) FROM scada_telemetry;"` |
| `ValidationError: Extra inputs are not permitted` | 往 state 里塞了未声明的键（如已删除的私有字段） | 对照 `src/schemas/state.py` |
| 描述长度断言失败 | 大模型输出波动 | 见 §3.12，检查提示词的长度预算是否需要再收紧 |
| 图构建类用例 skip | 没配 `DEEPSEEK_API_KEY` | `eval "$(grep -E '^export DEEPSEEK_API_KEY=' ~/.bashrc)"` |

> `tests/test_pipeline.py` 是**脚本式**的（`python -m` 跑），不是 pytest 用例；
> 它输出 30 行逐用例核对表 + 汇总的 `通过: N/30`。

---

## 2. 完整数据流（用测试用例 **A2** 逐步走一遍）

**用例 A2**：`PUMP-IS100-80-160-01`，`2026-09-13 07:30:00 ~ 08:09:55`（轴承干磨故障周期，480 点）

### 📊 全局一览：数据在每一步被"降维"

| 阶段 | 谁在算 | 数据形态 | 规模 |
| :--- | :--- | :--- | :--- |
| 0 输入 | Step 1 | 3 个字段 | 3 个值 |
| 1 取数 | Python / MySQL | `list[dict]`，11 列 | **480 条 ≈ 5280 个数字** |
| 1a 切段 | Python | 状态段列表 | **480 点 → 2 段** |
| 1b 算指标 | Python | 每段 21 个统计量 | **2 × 21 = 42 个数字** |
| 1c 规则判定 | Python | 规则码（机器码） | **5 种**（去重） |
| 1d 报警码 | Python | 去重集合 | **4 个码** |
| 2 语义化 | **大模型** | 1~6 条按 type 分类的现象描述 | **约 400~600 字** |
| 3 输出 | 图回流 | **data 盒子**（6 个子字段） | 1 个顶层键 |

> **大模型只在阶段 2 出现，且拿到的全是算好的数字，碰不到原始数据。**
> 阶段 1 的四小步全在一个节点（`analyze`）里跑，**原始遥测只活在函数局部变量里**。

---

### 阶段 0 ｜输入：Step 1 写进父图 `DiagnosisState`

```json
{
  "device_id": "PUMP-IS100-80-160-01",
  "start_time": "2026-09-13T07:30:00",
  "end_time": "2026-09-13T08:09:55"
}
```

只有 3 个字段。**窗口由 Step 1 推导**（通常是"报警时刻前推 N 分钟"），Step 2 不猜窗口。

> `alarm_code` 仍然在公共契约里（Step 3 的视觉提示词要用），但 **Step 2 不读它**（见 §3.14）。

---

### 阶段 1a ｜`analyze` 取数：一条 SQL 拿回原始遥测

调 `query_scada_telemetry()`，返回 `list[480 个 dict]`，每个 11 个键：

```jsonc
// 首条：故障刚起步
{"timestamp":"2026-09-13T07:30:00","flow_rate":100.05,"press_out":0.314,"press_in":0.023,
 "temp_de":45.5,"temp_nde":43.4,"vib_rms_de":1.58,"vib_rms_nde":1.41,
 "motor_current":23.09,"operating_state":"DEGRADING","alarm_code":"NONE"}

// 末条：温度已越停机线
{"timestamp":"2026-09-13T08:09:55","flow_rate":100.82,"press_out":0.311,"press_in":0.019,
 "temp_de":85.5,"temp_nde":68.1,"vib_rms_de":4.95,"vib_rms_nde":4.29,
 "motor_current":22.34,"operating_state":"WARNING","alarm_code":"TAHH-101;VAHH-102"}
```

**`temp_de` 从 45.5 → 85.5℃ 就是这条数据的主线。**

> 这 480 条**不进 state**：它们只在 `analyze_node` 的局部变量里存在，
> 函数返回时只交出"压缩后的小结果"。这是 v2 精简的核心收益（见 §3.15）。

---

### 阶段 1b ｜切段：把 480 点压成 2 段

```python
df['_seg_id'] = (df['operating_state'] != df['operating_state'].shift()).cumsum()
```

```
480 点 → 2 段
  段1 [DEGRADING] 07:30:00 ~ 08:00:00   361 点  1800s
  段2 [WARNING  ] 08:00:05 ~ 08:09:55   119 点   590s
```

---

### 阶段 1c ｜`_segment_metrics`：每段算 21 个统计量

以**段 2（WARNING）**为例，21 个键**全部列出**（数值取自本窗口实跑结果）：

```jsonc
{
  // —— 段定位（4）——
  "start": "2026-09-13T08:00:05",   // 该段起点时刻
  "end": "2026-09-13T08:09:55",     // 该段终点时刻
  "duration_sec": 590,              // 段持续秒数
  "data_points": 119,               // 段内帧数（5s 一帧）

  // —— 水力（5）——
  "avg_flow": 99.97,                // 流量均值 m³/h（额定 100）
  "avg_press_out": 0.312,           // 出口压力均值 MPa（额定 0.312）
  "avg_press_in": 0.02,             // 入口压力均值 MPa（额定 0.020，过低有汽蚀风险）
  "cv_flow": 0.51,                  // 流量变异系数 %（整段波动程度）
  "cv_press": 0.59,                 // 出口压力变异系数 %

  // —— 温度（3）——
  "slope_temp_de": 1.529,           // 驱动端温度斜率 ℃/min ← 关键
  "max_temp_de": 85.5,              // 驱动端温度峰值 ℃（70 预警 / 80 停机）← 关键
  "max_temp_nde": 68.1,             // 非驱动端温度峰值 ℃

  // —— 振动（3）——
  "slope_vib_de": 0.208,            // 驱动端振动斜率 mm/s 每分钟 ← 关键
  "max_vib_de": 5.12,               // 驱动端振动峰值 mm/s（3.5 上限 / 4.5 国标）← 关键
  "max_vib_nde": 4.41,              // 非驱动端振动峰值 mm/s

  // —— 电气（2）——
  "avg_motor_current": 22.51,       // 电机电流均值 A（额定 22.5）
  "max_motor_current": 23.08,       // 电机电流峰值 A（≥29 判满载跳闸）

  // —— 2 分钟滑窗 CV 峰值（2）——
  "cv_flow_peak": 0.62,             // 滑窗内流量 CV 峰值 %（捕"局部剧烈波动"）
  "cv_press_peak": 0.69,            // 滑窗内出口压力 CV 峰值 %

  // —— 拐点（1）——
  "ramp_start_time": "2026-09-13T08:01:00",  // 温度 1min 滑窗斜率首次 ≥0.8 ℃/min 的时刻

  // —— 状态（1）——
  "state": "WARNING"                // 该段的状态标签
}
```

> **谁在用**：17 个渲染进大模型的 `phases_text`，16 个参与 `_judge_flags` 规则判定，
> 两者重叠 12 个 —— **21 个全部有人读，没有一个是白算的**（有脚本可复核）。
>
> ★ 为什么只有**驱动端**有斜率（`slope_temp_de` / `slope_vib_de`）：
> 非驱动端斜率的两个字段（原 `slope_temp_nde` / `slope_vib_nde`）全项目没有任何地方读，
> 2026-09-17 删除；非驱动端仍保留峰值。同一天还删掉了 `inflection_time` /
> `inflection_value`（"温度首次达到预警线的时刻与温度值"）—— 同样零读者，
> 且"首次越预警线"已被 `max_temp_de` + 阈值判定覆盖。
> 那次精简把每段 25 → **21** 个统计量。

---

### 阶段 1d ｜`_judge_flags`：统计量 → **规则码**

**判据一个字都没改，变的只是"输出什么形状"**（2026-09-20 起输出机器码）：

```
窗口级（进 state 的 data.threshold_flags）—— 去重、按首次触发顺序
['BEARING_TEMP_DE_RAMP_SLOW', 'BEARING_TEMP_DE_TRIP',
 'BEARING_TEMP_DE_RAMP_SHARP', 'VIB_DE_TRIP', 'VIB_NDE_WARN']

段级（每个 phase 自带 rule_hits，"哪一段出了什么问题"一目了然）
[DEGRADING段] ['BEARING_TEMP_DE_RAMP_SLOW']
[WARNING段]   ['BEARING_TEMP_DE_TRIP', 'BEARING_TEMP_DE_RAMP_SHARP', 'VIB_DE_TRIP', 'VIB_NDE_WARN']
```

**中文事实句没有消失，只是挪到了"拼提示词的那一刻"**（`render_rule_hits`，只进 prompt、不进 state）：

```
- [DEGRADING段 07:30:00~08:00:00] 驱动端温度缓慢劣化（BEARING_TEMP_DE_RAMP_SLOW），约从 07:30:55 开始
- [WARNING段 08:00:05~08:09:55] 驱动端温度超停机线（BEARING_TEMP_DE_TRIP）
- [WARNING段 08:00:05~08:09:55] 驱动端温度急剧恶化（BEARING_TEMP_DE_RAMP_SHARP），约从 08:01:00 开始
- [WARNING段 08:00:05~08:09:55] 驱动端振动超国标停机线（VIB_DE_TRIP）
- [WARNING段 08:00:05~08:09:55] 非驱动端振动超良好区上限（VIB_NDE_WARN）
```

**为什么要这么分**：state 只留短而稳定的**码**（下游 Step 6 / 报告可直接吃，改文案不影响它们）；
人读的中文只在提示词里生成一次。"约从 08:01:00 开始"这类上下文由码表的
`extra_field` / `extra_template` 声明，渲染时现取——所以 `ramp_start_time` 依然有用。

**看第 2 条和第 3 条**：同样是温度越线，走了两条不同判据 —— 温度绝对值（85.5 ≥ 80℃）与
斜率（1.53 ℃/min ≥ 0.8），阈值全部来自 `rules/thresholds.py`，代码里没有魔法数字。
码表 `RULE_CATALOG`（**17 条**）也放在同一个文件里，与 `_judge_flags` 的分支一一对应，
有测试守着不许漂移（见 §5「加用例」前的防漂移说明）。

---

### 阶段 1e ｜报警码：只认窗口数据

```python
# 从这一批帧的 alarm_code 列提取 → 去重排序
effective_alarm_codes = 'TAH-101;TAHH-101;VAH-102;VAHH-102'
```

写进 `data.alarms` 三件套（`effective` / `last` / `all`）—— 这是报警码**唯一**的出处。
（v1 里它还有两个兄弟字段 `last_alarm_codes` / `all_alarm_codes_in_window`，
以及一份挂在顶层的重复副本，都已在 v2 删除。）

---

### 阶段 1f ｜`overall` 全局概要

```jsonc
{"total_points":480, "start_state":"DEGRADING", "end_state":"WARNING",
 "has_shutdown":false, "phase_count":2,
 "effective_alarm_codes":"TAH-101;TAHH-101;VAH-102;VAHH-102",
 "max_temp_de_overall":85.5, "max_vib_de_overall":5.12}
```

---

### 阶段 2a ｜送进 Prompt 的三份材料

| # | 材料 | 规模 | 内容 |
| :---: | :--- | :--- | :--- |
| ① | `overall_summary` | 9 行 | 点数 / 窗口 / 状态 / **窗口内出现过的报警码** / 峰值 |
| ② | `phases_text` | 2 段 × 8 行 | 每段的指标表 |
| ③ | `flags` | 5 条 | 渲染后的规则事实（中文标签 + 码 + 段定位，见 1d） |

**注意：送进去的全是算好的数字，没有一行原始遥测。**

---

### 阶段 2b ｜大模型返回（经 Pydantic 校验）

**1~6 条按 type 分类的数据现象描述**（A2 窗口实测：5 条，合计 407 字）：

```
[1] type=TREND       evidence=[slope_temp_de, max_temp_de, BEARING_TEMP_DE_RAMP_SLOW, …]
    驱动端温度全程单调爬升，07:30 起缓慢劣化（slope≈0.75 ℃/min），08:00 后转为急剧恶化……
[2] type=THRESHOLD   evidence=[max_temp_de, BEARING_TEMP_DE_TRIP]
    驱动端温度越过停机线，命中 BEARING_TEMP_DE_TRIP，出现在 08:00:05~08:09:55 的 WARNING 段。
[3] type=THRESHOLD   evidence=[max_vib_de, VIB_DE_TRIP, VIB_NDE_WARN]
    驱动端振动越过国标停机线（峰值 5.12 mm/s），非驱动端振动同时越过良好区上限……
[4] type=CORRELATION evidence=[slope_temp_de, slope_vib_de, …]
    08:00 前后驱动端温度与驱动端振动同步加速上升……
[5] type=OTHER       evidence=[avg_flow, cv_flow_peak]
    流量与出口压力全程平稳（avg≈100 m³/h、0.312 MPa，CV<1%），未发生停机，窗口内无报警码。
```

**关键点**：描述里的每个数字都能在阶段 1 的统计量里找到出处；**`evidence` 里的名字也必须是材料里真实存在的**
（节点侧有白名单过滤，大模型编造的名字会被丢掉）。大模型做的是**翻译**，不是**计算**。

> ⚠️ 报警码只按"窗口内出现过"讲，**不能给它安时间** ——
> 材料里没有单条报警码的起止时刻（见 §3.14），硬凑时间就是编造。

---

### 阶段 3 ｜输出：只回写 data 一个盒子

```
子图 state（= 公共契约 DiagnosisState，9 个盒子）：
  [输入] state.context : trace_id / device_id / start_time / end_time / alarm_code / image_refs
  [产出] state.data    : DataState
           ├ telemetry_ref    取数来源（source / trace_id / 设备 / 窗口）
           ├ quality          PENDING | OK | EMPTY | PARTIAL（+ total_points / reason）
           ├ metrics          {"overall": {...}, "phases": [...]}
           ├ threshold_flags  规则码（去重，按首次触发顺序）
           ├ alarms           AlarmState{effective, last, all}
           └ descriptions     list[DataDescription]（当前 1 条 type=OTHER，第 3 步改多条分类）
  ────────────────────────────────────────────
  没有私有字段：480 条原始遥测一条都没进 state
```

---

## 3. 关键设计决策与理由

### 3.1 为什么要分段？

**因为原始数据量级根本没法直接用。**

24 小时窗口是 **17280 个点**。无论给大模型还是给下游 Step，逐帧数据都没有意义 —— 有价值的是"**它经历了哪几个阶段、每个阶段什么样**"。

分段就是**降维的第一步**：17280 点 → 10 段，然后每段只用 21 个统计量代表。

### 3.2 为什么按 `operating_state` 分段，而不是按数值突变点？

| | 按 `operating_state` 切 | 按数值突变切 |
| :--- | :--- | :--- |
| 分段边界 | 与业务语义对齐（"什么时候从 DEGRADING 变成 WARNING"） | 需要自己定义突变阈值 |
| 下游可读性 | 高，Step 5/7 一眼看懂状态演变 | 低，得到的是"第 137 点斜率变了" |
| **代价** | **依赖 SCADA 侧的状态标签** | 不依赖标签 |

**选它的理由**：这条链路里 `operating_state` 是 SCADA 侧已标注好的业务语义，直接用它切段，边界天然与实际工况一致；而且下游（尤其 Step 7 报告）要展示的正是"状态演变"。

**但要清楚它的代价**（见 §6 已知边界）：分段正确性**依赖上游标签正确**。如果某天 DCS 不推这个字段，或标签标错，分段就会错 —— 而系统不会报错。

### 3.3 为什么停机段要跳过告警判定？

```python
# rules/thresholds.py
ACTIVE_STATES = frozenset({"NORMAL", "DEGRADING", "WARNING",
                           "CRITICAL_CAVITATION", "UNBALANCE_MISALIGNMENT"})
# TRIP_SHUTDOWN 不在其中 → 该段不做流量/电流/振动判定
```

**因为停机时流量、电流、振动全都归零。** 若照常判定，会立刻报出一堆"流量低""电机欠载"的假告警。

**停机是故障的结果，不是新的异常源。** 跳过它是刻意的。

实测这道闸门确实拦住了东西 —— 停机段的滑窗 CV 峰值是 **464.95%**，
远超 12% 的 CV 判据。若不排除，这里会凭空报出"波动超离散度阈值"。

> **这里有第二道、且不依赖标签的闸门**：`FLOW_ACTIVE_THRESHOLD_M3H = 20.0`。
> 流量低于 20 m³/h 直接视为停机段，不参与流量偏离 / CV 告警 —— 这道判据只看物理量。
>
> 所以准确的说法是：`ACTIVE_STATES` 是**兜底白名单**，用途是排除 `TRIP_SHUTDOWN` 这个
> 物理上没有意义的工况段；**规则本身的主判据仍然只依赖物理量**。
> 但这毕竟读了标签，标签错了这道兜底也会跟着错 —— 记在 §6。

### 3.4 为什么阈值全部集中在 `rules/thresholds.py`？

三个理由：

1. **同一物理量在不同标准里限值不同**（轴承温度：GB 50275 是 80℃，IOM 文本写 85℃），必须显式标注"采用了哪个、为什么"；
2. **Step 2 的规则判定与 Step 6 的安全门禁共用同一批阈值** —— 放两处必然漂移，一处调了另一处没调，两份结论就打架；
3. 每条判定结论都要能回填出处，供报告 100% 溯源。

现在 `analyzer.py` 里**没有任何规范级的魔法数字**（有测试守护这条）。

### 3.5 为什么状态用 Pydantic，而不是 `TypedDict`？

`TypedDict` 只是个类型提示，**运行时不校验**。字段名写错、节点回写了一个没声明的键，它都静默通过。

换成 Pydantic（`extra="forbid"`）之后：

- 字段名写错 → 当场报错
- 类型不对 → 当场报错
- 回写了未声明的键 → 当场报错（v2 删掉私有字段后，这条还顺带保证了节点不会偷偷往 state 里塞大数据）

**代价**：LangGraph 传给节点的是**模型实例不是 dict**，所以 `DiagnosisState` 上补了下标访问协议（`__getitem__`），让原有节点代码一行都不用改。

### 3.6 子图 state 与公共契约的关系（langgraph 1.2.11 实测）

| 规则 | 含义 |
| :--- | :--- |
| 子图能读到的键 | = **子图 schema 里声明了的键** |
| 能回流父图的键 | = **父图 schema 里也有的键** |
| 子图独有的键 | = 私有，**不会**回流父图 |

所以**不能"子图只定义私有字段"** —— 那样子图连 `device_id` 都读不到（实测确认）。

**Step 2 当前的做法**：因为已经没有任何私有字段，子图直接拿公共契约
`DiagnosisState` 当 state，`data_state.py` 随之删除。
将来哪个子 Agent 真的需要私有字段，就按"**继承公共契约 + 补私有字段**"扩展。

> **"私有 ≠ 免费"**：私有字段虽然不外流，**仍然留在子图 state 里**，
> 子图内部每走一步照样要被合并一次。这就是 v2 把取数与计算合并、让原始帧
> 走函数局部变量的原因（§3.15）。

### 3.7 为什么取数函数不用 `@tool` 装饰器？

`@tool` 是**给大模型绑定工具**用的（让 LLM 决定何时调用）。

而 `query_scada_telemetry` 是**节点里的固定步骤**，由 Python 代码直接调用，大模型根本决定不了。给它套 `@tool` 只会：

- 让调用方式变成 `xxx.invoke({...})`（多一层没必要的包装）
- 让人误以为"这是给 LLM 用的工具"

### 3.8 为什么报警码要"两个来源"？现在为什么只剩一个？

v1 里报警码有两条通道：

| 通道 | 来源 | 特点 |
| :--- | :--- | :--- |
| 报警事实 | SCADA 的 `alarm_code` 列 | 权威 —— SCADA 已按持续时长确认过 |
| 规则求值 | `_judge_flags` 扫统计量 | 补盲 —— 能发现 SCADA 没报的越限 |

**v2 把它收敛成一条线**：报警码只从窗口数据提取（`overall.effective_alarm_codes`），
`_judge_flags` 继续负责"报警表没列的判据"：两端振动差异、流量偏离额定值、
**出口压力偏离额定值**、CV 离散度、入口压力。

理由：SCADA 的 `alarm_code` 本来就是用**同一批阈值**算出来的
（`scripts/scada_generator.py` 的 80/70/4.5/3.5/85/-0.040/29/15 与 `rules/thresholds.py` 同源），
两套"越限事实"高度重叠；而 `_judge_flags` 独有的那部分（带实测值与阈值、可溯源）才是它不可替代的价值。

### 3.9 为什么 `temperature=0`、`max_tokens=4096`？

- **`temperature=0`**：工业诊断要求"同一窗口、同一数据、同一结论"。0.7 会让同一窗口的描述长度在 765~1031 字之间乱跳，**字数断言必然时好时坏**。（实测：改成 0 后波动收窄到 ±50 字）
- **`max_tokens=4096`**：原来是 1024。结构化输出要把 JSON 骨架 + 正文一起塞进这个额度，**密集故障窗口会超出被截断**，`with_structured_output` 解析失败返回 `None`，节点直接崩。改成 4096 后解决。

### 3.10 为什么 Prompt 里不写 JSON 格式？

因为用了 `with_structured_output(SemanticizeOutput)` —— **schema 由 LangChain 自动注入**。

提示词里再手写一遍 JSON 格式，既冗余又容易和 schema 冲突。现在提示词**只讲业务规则**（写什么、不写什么），格式交给 Pydantic。

同时这也带来了强约束：LLM 漏字段、类型写错，**直接抛 `ValidationError`**，而不是像 `JsonOutputParser` 那样静默返回脏数据。

### 3.11 为什么 `descriptions` 是 1~6 条，而不是一段话？

组长的契约把 Step 2 的语义化产出定义成 `data.descriptions: list[DataDescription]`
（每条 `{type, description, evidence}`），所以这里**不是**"一段 llm_description"：

- **按类型分条**，下游才能按需取用（Step 4 RAG 检索、Step 7 报告分节）；
  一段整话既没法检索、也没法按类别筛选。
- **type 六选一**（TREND / ANOMALY / THRESHOLD / CORRELATION / ALARM / OTHER），
  由 Pydantic 的 `Literal` 钉死 —— 大模型写错类别直接 ValidationError。
- **条数 1~6**：上限定 6 是防大模型"刷条数"把描述拆成流水账；下限定 1 是要求
  即使全窗口平稳也必须给出结论，不能返回空列表。
- **evidence 必须能回溯**：节点侧拿"材料里真实出现过的指标名 + 规则码"做白名单，
  把编造的（如 `bearing_temperature`）过滤掉再写进 state。

> v1 的另两个字段（`basic_judgment` / `rag_search_queries`）仍然没有回来：
> 前者与描述重叠，后者是 Step 4 的活；组长的契约里也没有它们。

### 3.12 为什么描述长度预警线按**事件数**算？

原先的规则是 `≤5 段 → 500 字；>5 段 → 1000 字`。问题是 A2 这种窗口段数少、事件却不少，会被误报。

**改成按事件数**。v1 用的是「状态切换次数 + 报警段数」；报警事件查询删掉后，
"报警段数"没有了，改用**窗口内出现的报警码种数**作为"有多少种报警在响"的近似度量：

```python
warn_threshold = min(1000, 500 + 40 × 状态切换次数 + 20 × 报警码种数)
```

> 这条只是**软预警**（打印一句提醒），不影响任何断言。真实跑批后可再按实测调。

### 3.13 为什么 `summarize` 节点要显式处理"解析失败返回 None"？

`with_structured_output` 在解析失败时返回 `None`（最常见原因是输出被 `max_tokens` 截断）。
如果不处理，下游会在访问返回值的某个字段时抛出莫名其妙的 `'NoneType' object has no attribute ...`。
所以节点里显式检查并给出一句能直接定位问题的报错。

### 3.14 ★ 为什么删掉报警事件查询（`query_alarm_events`）？

它原本在 `summarize` 节点里**把同一个窗口的行再查一遍库**，产出"一次连续报警 = 一条"
的游程编码（带首末时间、持续时长、报警时设备状态、段内峰值），再渲染成 prompt 材料。

删掉的理由：

1. **信息几乎全部重复**：它要的那批行，`analyze` 节点刚取过一遍。同一个窗口查两次库，
   24h 窗口就是白白多扫 1.7 万行。
2. **它的唯一读者已经不需要它**：报警段的"持续多久"这条信息，`phases`（按
   `operating_state` 分的状态段）已经近似回答了 —— 而 `operating_state` 本来就是由
   `alarm_code` 反推的。所以"有报警的那段时间从哪到哪、期间温度振动爬到多少"这条线还在，
   只是粒度从"单条报警码"变成"状态段"。
3. **连带省掉一整套防御代码**：游程编码、10s 去抖、30 段上限保护，约 150 行。

**代价（必须记住）**：大模型不再知道**单条报警码的精确起止时刻与持续时长**。
所以提示词里原来那句"状态切换时刻必须提及：写出切换时间**与触发的报警码**"
已拆成两句话 —— 否则大模型会把报警码硬凑到某个时间点上（那就是幻觉）。

### 3.15 ★ 为什么取数和确定性计算要合成一个节点？

v1 是 `fetch_data` → `calculate_metrics` → `semanticize` 三节点，
取回的数据必须写进 state 才能交给下一个节点，于是 24h 窗口的 **1.7 万条原始遥测
每走一步都要被合并拷贝一次**（v1 的 §6 已知边界 #5 就记着这件事）。

合并成一个 `analyze` 节点后，原始帧只是**函数局部变量**，返回时只交出压缩后的小结果：

| | v1 | v2 |
| :--- | :--- | :--- |
| 节点 | 3 | **2** |
| 私有字段 | 2（1.7 万条帧 + 18 段报警） | **0** |
| 每窗口数据库查询 | 2 次 | **1 次** |
| Step 2 产出公共字段 | 8 | **3** |

### 3.16 ★ 为什么报警码不再读用户入参的 `alarm_code`？

一个数据只能有一个真相来源。v1 里 `alarm_code` 入参是"窗口内没有任何报警帧时的兜底"，
结果是同一次分析可能给出两种报警码（数据里的 vs 用户嘴里的）。

v2 起 Step 2 **完全不读** `alarm_code`：窗口里没有报警就是 `NONE`。

> 字段本身**还留在公共契约里** —— Step 3 的视觉提示词仍在用它
> （`vision_client.py` 的 `alarm_code` 占位符）。
> 将来若要让它也改读 Step 2 的产出，是 Step 3 那边的独立改动。

### 3.17 ★ 出口压力偏离判据（2026-09-17 补全的"半成品判据"）

`rules/thresholds.py` 早就定义了 `DEV_PRESS_PCT = 20.0`（规范 5.1：`|Dev_press| > 20%` → 水力压头异常），
但 `_judge_flags` 一直没实现它 —— 全项目只有一条测试断言引用过这个常量。现已补上：

```python
dev_press = (ph["avg_press_out"] - RATED_PRESS_OUT_MPA) / RATED_PRESS_OUT_MPA * 100
if abs(dev_press) > DEV_PRESS_PCT:
    flags.append(f"{tag} 出口压力工况偏离额定值 >{DEV_PRESS_PCT:g}% (实际 {dev_press:+.1f}%)")
```

**它跟流量判据共用同一道闸门**（`avg_flow > FLOW_ACTIVE_THRESHOLD_M3H = 20`）。
为什么必须加这道闸门，实测数据一目了然：

| 段 | avg_flow | avg_press_out | 压头偏离 | 结论 |
| :--- | ---: | ---: | ---: | :--- |
| 重启欠载段（泵在转） | 42 m³/h | 0.131 MPa | **-58%** | ✅ 报 —— 真阳性，扬程只有额定的 42% |
| 停机后刚重启的瞬间 | 0.9 m³/h | 0.005 MPa | -98% | ❌ 不报 —— 流动还没建立，压力低是必然 |
| 停机段 | 0.04 m³/h | 0.001 MPa | -99% | ❌ 不报 —— 本来就被 `ACTIVE_STATES` 白名单排除 |

⚠ 在当前数据集里这条判据**只在重启欠载段触发**，其余时间出口压力紧贴额定的 0.312 MPa。
防护：B3 / C5 两个"重启过渡"用例加了 `alarm_contains: ["PRESS_OUT_DEV"]`，
另有一条**不依赖数据库**的单测（`test_pressure_deviation_rule_fires_and_shares_the_flow_gate`）
同时验证"要报"与"被闸门挡住"两种情况。

---

## 4. 测试用例清单（30 个）

跑法：`python -m tests.test_pipeline` → 期望 `通过: 30/30`

### 分组与意图

| 组 | 用例 | 覆盖意图 |
| :--- | :--- | :--- |
| **A 基础**（4） | A1~A4 | 单段正常 / 完整故障周期 / 只截 WARNING / 只截 TRIP |
| **B 故障形态**（7） | B1~B5, B4S, E5S | 高温越线、阈值抖动、重启欠载、24h 真实密度、高碎片密度 |
| **C 窗口边界**（6） | C1~C6 | 单点 / 两点 / 1 分钟 / 停机段 / 启动过渡 / **未知设备降级** |
| **D 数据一致性**（4） | D1~D4 | 三个历史数据 Bug 的守护 + 跨状态边界切段 |
| **E 报警码来源**（5） | E1~E5 | 报警码只认窗口数据；入参不再被采用 |
| **G 状态机**（2） | G1, G2 | 无切换 / 多状态切换 |
| **H 碎片**（2） | H1, H2 | 切段粒度在碎片场景下的表现 |

### 三个值得单独说的用例

**① A2 —— 完整故障周期**（本文档全程用的例子）
```
窗口 2026-09-13T07:30:00 ~ 08:09:55
期望 points=480, phases=2, alarm_count≥4
     states_contains=[DEGRADING, WARNING]
     effective_alarm_codes_contains=[TAH-101, TAHH-101, VAH-102, VAHH-102]
     alarm_contains=[BEARING_TEMP_DE_TRIP, BEARING_TEMP_DE_RAMP_SHARP, VIB_DE_TRIP]
```

**② H1 —— 碎片切分**（暴露"按状态切段"的粒度问题）
```
窗口 10:55:10 ~ 10:55:40（只有 7 个点）
实际切成 4 段：NORMAL → WARNING → NORMAL → WARNING
```
这是"按 `operating_state` 切段"的**直接后果** —— 数值在阈值附近抖动时，
状态会频繁翻转，产生大量极短段。用例把它**固定成已知行为**，而不是假装不存在。

**③ E2 —— 入参不再被采用**（v2 的行为变更）
```
窗口 T1 开机后纯 NORMAL 段（60 点），入参 alarm_code = "TAHH-101;VAHH-102"
期望 effective_alarm_codes = "NONE"
```
窗口里没有报警就是 NONE —— 报警码只有一个真相来源（见 §3.16）。

---

## 5. 怎么新增/修改测试用例

用例定义在 `tests/test_pipeline.py` 的 `cases` 列表里，每个是一个 7 元组。
下面是 **A2 的原文**：

```python
("A2", "完整故障周期（DEGRADING→WARNING）",   # ① 用例 ID  ② 名称
 DEV, T1_DEG_START, T1_WARN_END,              # ③ 设备位号  ④ 起始  ⑤ 结束
 "",                                          # ⑥ 入参 alarm_code（Step 2 已不读，仅兼容签名）
 {"points": 480,                              # ⑦ 期望值字典
  "phases_min": 2, "phases_max": 3,
  "states_contains": ["DEGRADING", "WARNING"],
  "alarm_count_min": 4,
  "effective_alarm_codes_contains": ["TAH-101", "TAHH-101", "VAH-102", "VAHH-102"],
  "alarm_contains": ["BEARING_TEMP_DE_TRIP", "BEARING_TEMP_DE_RAMP_SHARP", "VIB_DE_TRIP"],
  "note": "温度 45→85.5℃ 越停机线，振动最高 5.12mm/s 越国标"}),
```

### 支持的期望字段（全部可选，缺省即不校验）

| 类别 | 字段 | 含义 |
| :--- | :--- | :--- |
| **精确** | `points` / `phases` / `alarm_count` / `states` | 完全相等 |
| **范围** | `points_min` / `points_max` / `phases_min` / `phases_max` | 上下界 |
| | `alarm_count_min` | **去重后规则码种数**的下界（**只有 `_min`，没有 `_max`**） |
| | `max_desc_len` / `min_desc_len` | LLM 描述字数区间 |
| **状态** | `states_contains` | **按序**子序列 |
| | `states_set_contains` | **无序**集合 |
| **报警码** | `effective_alarm_codes` | 精确匹配（取自 `data.alarms.effective`） |
| | `effective_alarm_codes_contains` / `..._excludes` | 包含 / 排除 |
| **规则码** | `alarm_contains` / `alarm_excludes` | 规则码子串包含 / 排除（★ 2026-09-20 起比对的是机器码，不再是中文句） |
| | `no_rule_hits_on_states` | 这些**状态**的段上不许有任何规则命中（替代原来的 `alarm_excludes: ["[NORMAL段]"]`） |
| **数据质量** | `quality_status` | `data.quality.status` 精确匹配（OK / EMPTY / PENDING / PARTIAL） |
| | `quality_reason_contains` | `data.quality.reason` 子串包含（降级原因就写在这儿） |
| **其他** | `has_alarm` | 是否有规则命中（等价 `alarm_count > 0`） |
| | `expect_data_bug` | 历史数据 Bug 是否已修 |
| | `note` | 说明，不参与校验 |

> v2 删除了 `last_alarm_codes` 这个期望字段（对应公共字段已从契约删除）。
> `alarm_count` 的口径也变了：从"告警行数"改为**去重后的规则码种数** ——
> 所以 B2 / B4S 这两个 56 段的抖动窗口从 5 降到了 **1**（始终只有 `FLOW_DEV` 一种码），
> 这不是漏报，是去重。

### 加用例的步骤

1. **先查数据**，别猜：
   ```sql
   SELECT operating_state, COUNT(*) FROM scada_telemetry
   WHERE device_id='PUMP-IS100-80-160-01'
     AND timestamp BETWEEN '...' AND '...' GROUP BY operating_state;
   ```
2. 在 `cases` 里加一条，`note` 写清这个窗口在测什么
3. 跑 `python -m tests.test_pipeline`，看实际值对不对得上

> **断言口径：绑物理量，不绑数据指纹。**
> ✅ `states_contains=["DEGRADING","WARNING"]`、`alarm_contains=["BEARING_TEMP_DE_TRIP"]`、
> `max_desc_len=900`、`effective_alarm_codes_contains=["TAHH-101"]`
> ❌ 绑死在某个窗口恰好切出几段、恰好有多少点
>
> **历史教训**：早期断言绑死了某次生成数据的点数/段数，同一套代码
> 在换了一批数据后跑出过 `26/27`、`9/30`、`27/30` 三种结果 —— 失败的是断言，不是代码。
> 所以现在窗口级用 `points` 这类精确断言，形态级一律用 `states_contains` / `alarm_contains`
> 这类**语义断言**。

---

## 6. 已知边界

| # | 边界 | 影响 | 现状 |
| :---: | :--- | :--- | :--- |
| 1 | **分段依赖 SCADA 状态标签** | 标签缺失/标错时，**分段**与"停机段跳过"都会错，且**不报错**（静默产出零告警） | 当前数据集的 `operating_state` 可靠。缓解：告警判定另有 `FLOW_ACTIVE_THRESHOLD_M3H=20` 这道**只看物理量**的闸门。长期应增加"标签与数值不一致"的核验告警 |
| 2 | **按状态切段会产生碎片** | 数值在阈值附近抖动时切成大量极短段（H1：7 点 4 段） | 已用 H1/H2 固定为已知行为；告警判定按段进行，碎片会让同一条告警重复出现 |
| 3 | **当前数据集缺两类故障场景** | 不含 `CRITICAL_CAVITATION`（气蚀）与 `UNBALANCE_MISALIGNMENT`（动不平衡），也没有 `PAL-103`（入口压力低） | 四大故障类型只覆盖了密封泄漏一条线；原 B1/B2/B3 用例已改用等效窗口，`note` 里注明。**要恢复覆盖需生成含相应场景的数据集** |
| 4 | **CV 判据分不清"爬坡"与"高频波动"** | 重启时流量从 0 爬升，2 分钟滑窗内被算成高频波动 → **误报"波动超离散度阈值"**。实测：T1 重启段 `CV_flow_peak=82.3%`、T2 重启段 `88.6%`（判据 12%） | **确认为已知误报**（重启段状态是 WARNING，在白名单内，挡不住）。停机段更极端（`464.95%`）但被白名单排除。根治需先**去趋势**再算 CV |
| 5 | **单条报警码的时序不再进 prompt** | 大模型只知道"窗口内出现过哪些报警码"，不知道每个码什么时候开始、持续多久 | v2 删掉报警事件查询的代价（§3.14）。需要精确时序时应从 `phases`（状态段起止）近似，或另外生成含报警时序的数据集 |
| 6 | **描述长度有大模型固有波动** | 同一窗口多次调用相差 ±50 字（`temperature=0` 也不能完全消除） | 字数断言已按实测上沿留余量 |
| 7 | **没有大模型 Key 就建不起图** | `data_nodes.py` 模块导入时就构造 LLM 客户端 → `pytest` 里涉及建图的用例只能 skip | 已知；改成节点内惰性构造可以根治，但那要连带改 Step 3 的导入方式，留待后续 |
| 8 | **Step 3 还没适配 9 盒子契约** | `vision_node` 仍按旧的扁平字段读写（`state.get("image_refs")`），在真实的 `DiagnosisState` 上会 `AttributeError`；`tests/test_vision_agent.py` 有 2 条 FAILED | 本轮范围外（用户决定 Step 3 推后）。Step 2 自身全绿：`tests/test_nodes.py` 27 passed + 端到端 30/30 |
| 9 | **本地 `src/orchestrator/graph.py` 还是旧主图** | 它仍挂着旧的 `vision_node`，跑起来会在 Step 3 崩；`build_diagnosis_graph`（组长的新主图）还没合过来 | 主图整合由组长走 PR；Step 2 的挂载用测试里的迷你父图验证（`test_mini_parent_graph_*`） |

---

## 7. 一句话总结

> **Python 负责"算什么"，大模型只负责"怎么说"。**
>
> 数据从 480 条原始遥测 → 2 个状态段 → 50 个统计量 → 5 条规则事实 → 4 个报警码
> → 一路降维，最后只把"算好的数字"交给大模型翻译成一段话。
> 大模型**碰不到原始数据**，也**写不出错数字**（它拿到的输入里根本没有可算的东西）。
>
> 而这一版做的事是：**让"接口"和"工序"也一起降维** ——
> 3 节点 → 2 节点、产出字段 8 → 3、私有字段 2 → 0、查库 2 次 → 1 次。
