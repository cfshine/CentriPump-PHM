# Step 2 · data_agent 实现说明

> **这份文档回答三件事**：
> 1. 数据从 Step 1 进来、到最终出去，**中间变成了什么形状**（用真实测试用例逐步展示）
> 2. 每个关键设计**为什么这么做**（尤其是"为什么按状态分段"这类问题）
> 3. **怎么提测**

---

## 0. 这个模块是什么

**一句话**：把 Step 1 递来的「一台设备 + 一个时间窗口」，变成一份结构化的工况分析结论，交给下游四个 Step。

```
                  ┌──────────────────────────────────────────┐
   Step 1 ───────►│  data_agent（Step 2）                     │
   4 个字段        │  fetch_data → calculate_metrics           │
                  │             → semanticize                 │
                  └────────────────┬─────────────────────────┘
                                   │ 只回写公共字段
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
   calculated_metrics        llm_description            threshold_flags
   （→ Step 5 因果归因）      （→ Step 4 RAG 检索）        （→ Step 6 安全门禁）
```

### 文件清单

| 文件 | 行数 | 职责 | 依赖 LangGraph？ |
| :--- | ---: | :--- | :---: |
| `data_state.py` | 42 | 子图状态：**继承**公共契约 + 补 2 个私有字段 | ❌ |
| `data_graph.py` | 25 | 子图定义（3 个节点 + 边） | ✅ 唯一 |
| `data_nodes.py` | 227 | 3 个节点适配函数 + **分段编排**（解包 → 切段 → 调用 → 回写） | ✅ |
| `data_tools.py` | 179 | 取数：逐帧遥测 + 报警段（游程编码） | ❌ |
| `repository.py` | 107 | `scada_telemetry` 表结构与 ORM | ❌ |
| `analyzer.py` | 233 | **段内**统计指标 + 规则判定（纯 pandas，核心算法） | ❌ |
| `timeseries_tools.py` | 216 | 滑窗 / 回归 / CV / 文本格式化工具 | ❌ |

> **`analyzer.py` 与 `timeseries_tools.py` 不含任何 LangGraph 与数据库依赖**，
> 只吃 DataFrame、只吐 dict/数值 —— 所以它们可以脱离图和数据库单独测试。
> 这是刻意的：算法归算法，编排归编排。

---

## 1. 怎么提测

### 1.1 环境准备

```bash
cd /home/yali_ai/work_file/LLM_Project/CentriPump-PHM

# 解释器：项目当前使用的 venv（已装 langgraph / langchain / pandas / sqlalchemy / pymysql）
PY=/home/yali_ai/work_file/LLM_test/day19_项目测试/step2_analysisAgent/backend/.venv/bin/python

# 大模型凭据：从系统环境变量读，代码里不做任何显式声明
eval "$(grep -E '^export DEEPSEEK_API_KEY=' ~/.bashrc)"

# 数据库：MySQL 127.0.0.1:3306 / scada_db（配置在项目根 .env）
```

### 1.2 两条提测命令

```bash
# ① 契约与挂载测试 —— 不需要数据库、不需要网络
$PY -m pytest tests/test_nodes.py -q
#    期望：9 passed
#    覆盖：状态分层契约 / 阈值唯一出处 / Step2 子图能挂进主图 / 私有字段不外流

# ② 端到端回归 —— 需要 MySQL + 大模型 Key
$PY -m tests.test_pipeline
#    期望：通过: 30/30
#    覆盖：30 个真实窗口，含故障周期、窗口边界、停机、重启、碎片、降级
```

### 1.3 命令失败自查

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| `ImportError: cannot import name 'model'` | `src/utils/llm_client.py` 不完整 | 确认文件末尾有 `model = init_chat_model(...)` |
| 全部用例 `points=0` | 数据库没起 / 窗口无数据 | `mysql -h127.0.0.1 -uroot -e "USE scada_db; SELECT COUNT(*) FROM scada_telemetry;"` |
| `ValidationError: Extra inputs are not permitted` | `.env` 里有 `Setting` 类未声明的键 | 对照 `.env.example` 与 `src/utils/config_loader.py` |
| 描述长度断言失败 | 大模型输出波动 | 见 §3.11，检查提示词的长度预算是否需要再收紧 |
| 挂载测试 skip | 没配 `DEEPSEEK_API_KEY` | `eval "$(grep -E '^export DEEPSEEK_API_KEY=' ~/.bashrc)"` |

> `tests/test_pipeline.py` 是**脚本式**的（`python -m` 跑），不是 pytest 用例；
> 它输出 30 行逐用例核对表 + 汇总的 `通过: N/30`。

---

## 2. 完整数据流（用测试用例 **A2** 逐步走一遍）

**用例 A2**：`PUMP-IS100-80-160-01`，`2026-09-13 07:30:00 ~ 08:09:55`（轴承干磨故障周期，480 点）

### 📊 全局一览：数据在每一步被"降维"

| 阶段 | 谁在算 | 数据形态 | 规模 |
| :--- | :--- | :--- | :--- |
| 0 输入 | Step 1 | 4 个字段 | 4 个值 |
| 1 取数 | Python / MySQL | `list[dict]`，11 列 | **480 条 ≈ 5280 个数字** |
| 2a 切段 | Python | 状态段列表 | **480 点 → 2 段** |
| 2b 算指标 | Python | 每段 25 个统计量 | **2 × 25 = 50 个数字** |
| 2c 规则判定 | Python | 中文告警 | **5 条** |
| 2d 报警段 | Python | 游程编码 | **18 段** |
| 3 语义化 | **大模型** | 1 段文字 + 3 字段 | **1 × 约 570 字** |
| 4 输出 | 图回流 | 12 公共 + 2 私有 | 14 个键 |

> **大模型只在阶段 3 出现，且拿到的全是算好的数字，碰不到原始数据。**

---

### 阶段 0 ｜输入：Step 1 写进父图 `DiagnosisState`

```json
{
  "device_id": "PUMP-IS100-80-160-01",
  "start_time": "2026-09-13T07:30:00",
  "end_time": "2026-09-13T08:09:55",
  "alarm_code": ""
}
```

只有 4 个字段。**窗口由 Step 1 推导**（通常是"报警时刻前推 N 分钟"），Step 2 不猜窗口。

---

### 阶段 1 ｜Node 1 `fetch_data`：取原始遥测

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

> ⚠️ 这 480 条是**私有字段**（`raw_telemetry_data`），只留在子图 state 里，**不会回流主图**。

---

### 阶段 2a ｜切段：把 480 点压成 2 段

```python
df['_seg_id'] = (df['operating_state'] != df['operating_state'].shift()).cumsum()
```

```
480 点 → 2 段
  段1 [DEGRADING] 07:30:00 ~ 08:00:00   361 点  1800s
  段2 [WARNING  ] 08:00:05 ~ 08:09:55   119 点   590s
```

---

### 阶段 2b ｜`_segment_metrics`：每段算 25 个统计量

以**段 2（WARNING）**为例：

```jsonc
{
  "start": "2026-09-13T08:00:05", "end": "2026-09-13T08:09:55",
  "duration_sec": 590, "data_points": 119,
  "avg_flow": 99.97, "cv_flow": 0.51,              // 水力
  "slope_temp_de": 1.529, "max_temp_de": 85.5,     // 温度 ← 关键
  "slope_vib_de": 0.208,  "max_vib_de": 5.12,      // 振动 ← 关键
  "avg_motor_current": 22.51, "max_motor_current": 23.08,
  "inflection_time": "2026-09-13T08:00:05",        // 首次越预警线的时刻
  "inflection_value": 70.5,
  "ramp_start_time": "2026-09-13T08:01:00",        // 斜率首次达"急剧恶化"的时刻
  "state": "WARNING"
}
```

---

### 阶段 2c ｜`_judge_flags`：统计量 → 中文告警

**输入是上面的统计量，输出 5 条**：

```
1. [DEGRADING段] 驱动端温度缓慢劣化 (斜率 0.75 ℃/min，黄色关注，约从 07:30:55 开始)
2. [WARNING段]   驱动端温度超停机线 (max 85.5 ≥ 80℃)
3. [WARNING段]   驱动端温度急剧恶化 (斜率 1.53 ℃/min ≥ 0.8，红色紧急，约从 08:01:00 开始)
4. [WARNING段]   驱动端振动超国标停机线 (max 5.12 > 4.5 mm/s)
5. [WARNING段]   非驱动端振动超良好区上限 (max 4.41 > 3.5 mm/s)
```

**看第 1 条和第 3 条**：同样是温度在升，DEGRADING 段斜率 0.75 判"缓慢劣化"，WARNING 段斜率 1.53 判"急剧恶化" —— 阈值 `0.2 / 0.8 ℃/min` 全部来自 `rules/thresholds.py`，代码里没有魔法数字。

---

### 阶段 2d ｜两个通道取报警码

**通道 A：从 `alarm_code` 列提取（SCADA 权威事实）**

```
effective_alarm_codes     = 'TAH-101;TAHH-101;VAH-102;VAHH-102'   ← 窗口内出现过（去重）
last_alarm_codes          = 'TAHH-101;VAHH-102'                   ← 最后一次非 NONE 的组合
all_alarm_codes_in_window = ['TAH-101','TAHH-101','VAH-102','VAHH-102']
```

**通道 B：`query_alarm_events` 取报警段（游程编码）**

```
TAH-101            | 08:00:05~08:02:40 | 持续155s | 温度(de)70.5→73.4℃(峰73.4) | 振动(de)2.84→3.12mm/s
TAH-101;VAH-102    | 08:02:45~08:02:45 | 持续  0s | …
TAH-101            | 08:02:50~08:04:15 | 持续 85s | …
…（共 18 段）
TAHH-101;VAHH-102  | 08:09:25~08:09:55 | 持续 30s | …
```

> 对比一下：**同样这个窗口，改造前返回 118 条**（每一帧有报警的记录各一条）。
> 现在 18 段 —— 因为一次连续报警只算一条，且带上了首末时间与持续时长。

---

### 阶段 2e ｜`overall` 全局概要

```jsonc
{"total_points":480, "start_state":"DEGRADING", "end_state":"WARNING",
 "has_shutdown":false, "phase_count":2,
 "max_temp_de_overall":85.5, "max_vib_de_overall":5.12}
```

---

### 阶段 3a ｜送进 Prompt 的四份材料

| # | 材料 | 规模 | 内容 |
| :---: | :--- | :--- | :--- |
| ① | `overall_summary` | 8 行 | 点数 / 窗口 / 状态 / 峰值 |
| ② | `phases_text` | 2 段 × 8 行 | 每段的指标表 |
| ③ | `alarm_events_section` | 18 行 | 报警段（见 2d） |
| ④ | `flags` | 5 条 | 阈值告警（见 2c） |

**注意：送进去的全是算好的数字，没有一行原始遥测。**

---

### 阶段 3b ｜大模型返回（经 Pydantic 校验）

```
llm_description（约 570 字）:
  07:30~08:00 设备处于 DEGRADING 段，流量约 100 m³/h、出口压力 0.312 MPa、
  电机电流约 22.5 A 均保持平稳，驱动端温度以 0.75 ℃/min 缓慢爬升，至 08:00
  前达 69.5 ℃……08:00:05 状态切换为 WARNING，触发 TAH-101（驱动端温度 70.5 ℃）……
  至 08:09:55，驱动端温度达全程峰值 85.5 ℃（超 80 ℃ 停机线），
  驱动端振动峰值 5.12 mm/s（超 4.5 mm/s 国标停机线）……

basic_judgment（约 150 字）: 一段话概括关键异常时刻

rag_search_queries: 3~5 条检索词
```

**关键点**：描述里的每个数字（100、0.312、22.5、0.75、69.5、85.5、5.12）**都能在阶段 2 的统计量里找到出处**。大模型做的是**翻译**，不是**计算**。

---

### 阶段 4 ｜输出：12 个公共字段回流主图，2 个私有字段留下

```
子图 state 共 14 个键：
  [公共] device_id / start_time / end_time / alarm_code
  [公共] calculated_metrics        dict(2)    ← overall + phases
  [公共] threshold_flags           list(5)
  [公共] effective_alarm_codes     str(33)
  [公共] last_alarm_codes          str(17)
  [公共] all_alarm_codes_in_window list(4)
  [公共] llm_description           str(~570)
  [公共] basic_judgment            str(~150)
  [公共] rag_search_queries        list(3~5)
  ────────────────────────────────────────────
  [私有] raw_telemetry_data        list(480)   ← 480 条原始遥测
  [私有] alarm_events              list(18)    ← 18 段报警

主图 state 收到 12 个键，全部属于公共契约: True
私有字段是否外流: ✅ 否
```

---

## 3. 关键设计决策与理由

### 3.1 为什么要分段？

**因为原始数据量级根本没法直接用。**

24 小时窗口是 **17280 个点**。无论给大模型还是给下游 Step，逐帧数据都没有意义 —— 有价值的是"**它经历了哪几个阶段、每个阶段什么样**"。

分段就是**降维的第一步**：17280 点 → 10 段，然后每段只用 25 个统计量代表。

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

实测这道闸门确实拦住了东西 —— 停机段（08:10~10:29）的滑窗 CV 峰值是 **464.95%**，
远超 12% 的流态失稳判据。若不排除，这里会凭空报出"疑似气蚀脱流"。

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
- 回写了未声明的键 → 当场报错

**代价**：LangGraph 传给节点的是**模型实例不是 dict**，所以 `DiagnosisState` 上补了下标访问协议（`__getitem__`），让原有节点代码一行都不用改。

### 3.6 为什么子图 state 要**继承**公共契约？

**实测结论**（langgraph 1.2.11）：

| 规则 | 含义 |
| :--- | :--- |
| 子图能读到的键 | = **子图 schema 里声明了的键** |
| 能回流父图的键 | = **父图 schema 里也有的键** |
| 子图独有的键 | = 私有，**不会**回流父图 |

所以**不能"子图只定义私有字段"** —— 那样子图连 `device_id` 都读不到（实测确认）。

正确做法是继承：公共字段**一处定义**，私有字段只加在子图。结果就是主图干净地拿到 12 个公共字段，480 条原始数据一条都没漏出去。

> **但"私有 ≠ 免费"**：私有字段虽然不外流，**仍然留在子图 state 里**，子图内部每走一步照样要合并一次。要彻底消除只能让它根本不进 state —— 那是后续优化项。

### 3.7 为什么取数函数不用 `@tool` 装饰器？

`@tool` 是**给大模型绑定工具**用的（让 LLM 决定何时调用）。

而 `query_scada_telemetry` / `query_alarm_events` 是**节点里的固定步骤**，由 Python 代码直接调用，大模型根本决定不了。给它们套 `@tool` 只会：

- 让调用方式变成 `xxx.invoke({...})`（多一层没必要的包装）
- 让人误以为"这是给 LLM 用的工具"

现在它们是普通函数，直接 `query_scada_telemetry(device_id, start, end)` 调用。

### 3.8 为什么报警事件要做**游程编码**？

`query_alarm_events` 的用途是「**查看报警发生时设备的状态**」。

而 5 秒采样下，同一个报警码会**连续出现几十上百帧**。如果每帧存一条：

| 窗口 | 每帧一条（旧） | 每次报警一段（新） |
| :--- | :---: | :---: |
| A2（10 分钟） | 118 条 | **18 段** |
| 24 小时 | 421 条 | **30 段** |

而且旧的形态**丢掉了最关键的信息**：这个报警持续了多久？

新形态每条带：`报警码 | 首末时间 | 持续时长 | 起始/结束状态 | 温度与振动的起止值和峰值 | 流量`。

**"TAH-101 从 08:00:05 持续到 08:02:40，期间温度从 70.5 升到 73.4℃"** —— 这才回答了"这次报警意味着什么"。

### 3.9 为什么还要**去抖**？

因为报警码会闪：

```
08:00:05  TAH-101     ← 报
08:00:10  NONE        ← 掉（温度在阈值上下抖）
08:00:15  TAH-101     ← 又报
```

不去抖的话，**一次持续 155 秒的报警会被切成两条**："1 帧的段 + 145 秒的段"，既看不出真实持续时间，又白占一条记录。

所以加了容忍度：相邻同码报警间隔 ≤ `ALARM_GAP_TOLERANCE_SEC`（10 秒 = 2 个采样周期）时视为**同一次**。合并后就是干净的 `08:00:05~08:02:40 持续 155s`。

### 3.10 为什么要**上限保护**？

报警码在阈值附近疯狂抖动时（比如 B2/B4S 那种碎片场景），可能产生上百段。不加限制会把 state 和提示词一起撑爆。

`MAX_ALARM_RUNS = 30`：超出时**按持续时长降序保留最长的**（抖动产生的短段信息量最低），其余**折叠成一条摘要**而不是静默丢弃：

```
[（折叠）] 另有 5 段短促报警未逐条列出（均为单帧触发）
```

### 3.11 为什么描述长度预警线按**事件数**算？

原先的规则是 `≤5 段 → 500 字；>5 段 → 1000 字`。

**问题**：A2 这个窗口只有 2 段，但**报警码升级了 18 次**，事件一点不少，却被 500 字卡住（实测写 545~650 字，一直被误报）。

**改成按事件数**：

```python
warn_threshold = min(1000, 500 + 40 × 状态切换次数 + 10 × 报警段数)
```

事件数才是"这段窗口里发生了多少事"的直接度量。A2 得 720、24h 得 1000，都按实测**上沿**留了余量。

> 另外说明：这条只是**软预警**（打印一句提醒），不影响测试通过。它的作用是提示开发者"描述是否没做好同类项合并"。

### 3.12 为什么 `temperature=0`、`max_tokens=4096`？

- **`temperature=0`**：工业诊断要求"同一窗口、同一数据、同一结论"。0.7 会让同一窗口的描述长度在 765~1031 字之间乱跳，**字数断言必然时好时坏**。（实测：改成 0 后波动收窄到 ±50 字）
- **`max_tokens=4096`**：原来是 1024。结构化输出要把 JSON 骨架 + 正文一起塞进这个额度，**19 个阶段的密集故障窗口会超出被截断**，`with_structured_output` 解析失败返回 `None`，节点直接崩。改成 4096 后解决。

### 3.13 为什么 Prompt 里不写 JSON 格式？

因为用了 `with_structured_output(SemanticizeOutput)` —— **schema 由 LangChain 自动注入**。

提示词里再手写一遍 JSON 格式，既冗余又容易和 schema 冲突。现在提示词**只讲业务规则**（写什么、不写什么），格式交给 Pydantic。

同时这也带来了强约束：LLM 漏字段、类型写错，**直接抛 `ValidationError`**，而不是像 `JsonOutputParser` 那样静默返回脏数据。

### 3.14 为什么报警码要"双通道"？

| 通道 | 来源 | 特点 |
| :--- | :--- | :--- |
| **报警事实** | SCADA 的 `alarm_code` 列 | **权威** —— SCADA 已按持续时长确认过 |
| **规则求值** | `_judge_flags` 扫统计量 | **补盲** —— 能发现 SCADA 没报的越限 |

两者**取并集**：宁可提示，不可漏报。

E 组用例专门验证这两个通道：E1 有入参+窗口内有、E2 有入参+窗口内无（回退到入参）、E3 无入参+窗口内有（自行提取）、E4 停机后清零。

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
| **E 报警码双通道**（5） | E1~E5 | 入参与窗口内报警的各种组合 |
| **G 状态机**（2） | G1, G2 | 无切换 / 多状态切换 |
| **H 碎片**（2） | H1, H2 | 切段粒度在碎片场景下的表现 |

### 三个值得单独说的用例

**① A2 —— 完整故障周期**（本文档全程用的例子）
```
窗口 2026-09-13T07:30:00 ~ 08:09:55
期望 points=480, phases=2, alarm_count≥4
     states_contains=[DEGRADING, WARNING]
     last_alarm_codes='TAHH-101;VAHH-102'
     alarm_contains=[温度超停机线, 温度急剧恶化, 振动超国标]
```

**② H1 —— 碎片切分**（暴露"按状态切段"的粒度问题）
```
窗口 10:55:10 ~ 10:55:40（只有 7 个点）
实际切成 4 段：NORMAL → WARNING → NORMAL → WARNING
```
这是"按 `operating_state` 切段"的**直接后果** —— 数值在阈值附近抖动时，
状态会频繁翻转，产生大量极短段。用例把它**固定成已知行为**，而不是假装不存在。

**③ C6 —— 未知设备降级**
```
device_id = "PUMP-UNKNOWN"（库里没有）
期望 points=0, phases=0, alarm_count=1, alarm_contains=["无数据"]
```
验证"窗口无数据是**正常响应**而不是异常"：节点返回结构完整的空结果 + 一条"无数据"告警，**不抛异常**。

---

## 5. 怎么新增/修改测试用例

用例定义在 `tests/test_pipeline.py` 的 `cases` 列表里，每个是一个 7 元组。
下面是 **A2 的原文**：

```python
("A2", "完整故障周期（DEGRADING→WARNING）",   # ① 用例 ID  ② 名称
 DEV, T1_DEG_START, T1_WARN_END,              # ③ 设备位号  ④ 起始  ⑤ 结束
 "",                                          # ⑥ 入参 alarm_code（可空）
 {"points": 480,                              # ⑦ 期望值字典
  "phases_min": 2, "phases_max": 3,
  "states_contains": ["DEGRADING", "WARNING"],
  "alarm_count_min": 4,
  "effective_alarm_codes_contains": ["TAH-101", "TAHH-101", "VAH-102", "VAHH-102"],
  "last_alarm_codes": "TAHH-101;VAHH-102",
  "alarm_contains": ["温度超停机线", "温度急剧恶化", "振动超国标"],
  "note": "温度 45→85.5℃ 越停机线，振动最高 5.12mm/s 越国标"}),
```

### 支持的期望字段（共 21 个，全部可选，缺省即不校验）

| 类别 | 字段 | 含义 |
| :--- | :--- | :--- |
| **精确** | `points` / `phases` / `alarm_count` / `states` | 完全相等 |
| **范围** | `points_min` / `points_max` / `phases_min` / `phases_max` | 上下界 |
| | `alarm_count_min` | 告警条数下界（**只有 `_min`，没有 `_max`**） |
| | `max_desc_len` / `min_desc_len` | LLM 描述字数区间 |
| **状态** | `states_contains` | **按序**子序列 |
| | `states_set_contains` | **无序**集合 |
| **报警码** | `effective_alarm_codes` | 精确匹配 |
| | `effective_alarm_codes_contains` / `..._excludes` | 包含 / 排除 |
| | `last_alarm_codes` | 最后一次非 NONE 的组合 |
| **告警文本** | `alarm_contains` / `alarm_excludes` | 子串包含 / 排除 |
| **其他** | `has_alarm` | 是否有告警（等价 `alarm_count > 0`） |
| | `expect_data_bug` | 历史数据 Bug 是否已修 |
| | `note` | 说明，不参与校验 |

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
> ✅ `states_contains=["DEGRADING","WARNING"]`、`alarm_contains=["温度超停机线"]`、
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
| 4 | **CV 判据分不清"爬坡"与"高频波动"** | 重启时流量从 0 爬升，2 分钟滑窗内被算成高频波动 → **误报"疑似流态失稳"**。实测：T1 重启段 `CV_flow_peak=82.3%`、T2 重启段 `88.6%`（判据 12%） | **确认为已知误报**（重启段状态是 WARNING，在白名单内，挡不住）。停机段更极端（`464.95%`）但被白名单排除。根治需先**去趋势**再算 CV |
| 5 | **私有字段仍占子图 state** | `raw_telemetry_data`（24h 达 1.7 万条）虽不外流，但子图内部每步仍要合并拷贝 | 彻底解决需让取数与计算合并成一个节点、数据走函数局部变量 |
| 6 | **描述长度有大模型固有波动** | 同一窗口多次调用相差 ±50 字（`temperature=0` 也不能完全消除） | 字数断言已按实测上沿留余量 |

---

## 7. 一句话总结

> **Python 负责"算什么"，大模型只负责"怎么说"。**
>
> 数据从 480 条原始遥测 → 2 个状态段 → 50 个统计量 → 5 条规则告警 → 18 段报警
> → 一路降维，最后只把"算好的数字"交给大模型翻译成一段话。
> 大模型**碰不到原始数据**，也**写不出错数字**（它拿到的输入里根本没有可算的东西）。
