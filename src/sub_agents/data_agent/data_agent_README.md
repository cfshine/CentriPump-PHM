# Step 2 · data_agent 实现说明

> **三句话导览**：
> 1. 输入是 `state.context`（设备 + 时间窗口），输出**只回写 `state.data`**（一个 `DataState` 盒子）；
> 2. 子图只有 **2 个节点**：`analyze`（取数 + 确定性计算）、`summarize`（大模型把数字说成人话）；
> 3. 判据与规则码表都在 `rules/thresholds.py`（唯一出处），代码里没有魔法数字。
>
> 契约版本：对齐组长 9 盒子契约（`origin/main`）；`.py` 共 6 个文件、约 980 行。

---

## 0. 这个模块是什么

**一句话**：把「一台设备 + 一个时间窗口」变成一份结构化的工况分析结论，装进 `state.data`。

**职责只有三件事**：

```
① 取 SCADA 数据  →  ② 确定性计算（描述状态，不做故障归因）  →  ③ 让大模型把数字说成人话
```

```
                    ┌──────────────────────────────────────────────┐
   state.context ──►│  data_agent（Step 2）                         │
   ├ device_id      │  analyze   取数 → 分段 → 统计 → 规则码 → 质量   │
   ├ start_time     │      ↓                                       │
   └ end_time       │  summarize 大模型 → descriptions（1~6 条）     │
                    └────────────────┬─────────────────────────────┘
                                     │ 只回写 data 一个盒子
                                     ▼
                          state.data（DataState）
   ┌───────────────┬──────────────┬──────────────┬──────────────┬────────────┐
   ▼               ▼              ▼              ▼              ▼            ▼
telemetry_ref   quality        metrics     threshold_flags   alarms   descriptions
（取数溯源）  （数据质量）  （overall+phases）（规则码）    （报警码）  （现象描述）
```

### 文件清单

| 文件 | 行数 | 职责 |
| :--- | ---: | :--- |
| `data_graph.py` | 47 | 子图拓扑（2 节点）+ 路线 B 入口 `run_data_agent` |
| `data_nodes.py` | 308 | 2 个节点函数、回写助手 `_data_update`、大模型输出契约 |
| `data_tools.py` | 60 | 取数：**一条 SQL** 按设备 + 窗口取帧 |
| `repository.py` | 107 | `scada_telemetry` 表结构与 ORM |
| `analyzer.py` | 248 | 段内统计指标 + 规则判定（纯 pandas，产出机器码） |
| `timeseries_tools.py` | 212 | 滑窗 / 回归 / CV / 报警码提取 / 规则码渲染 |

> `analyzer.py` 与 `timeseries_tools.py` **不含 LangGraph 与数据库依赖**，只吃 DataFrame、
> 只吐 dict/数值，可以脱离图和数据库单独测试。

---

## 1. 怎么提测

```bash
cd /home/yali_ai/work_file/LLM_Project/CentriPump-PHM
PY=/home/yali_ai/work_file/.venv/bin/python                    # 项目自身 venv
eval "$(grep -E '^export DEEPSEEK_API_KEY=' ~/.bashrc)"        # 大模型凭据（代码不显式声明）

$PY -m pytest tests/test_nodes.py -q          # ① 契约与节点行为（离线可跑）→ 期望 30 passed
$PY -m tests.test_pipeline                    # ② 端到端 30 用例（需 MySQL + Key）→ 期望 30/30
$PY -m pytest tests/ -q                       # 全量 → 期望 57 passed
```

> ★ `tests/` **有意不入库**（`.gitignore` 已忽略，文件只在本地）：这是本项目自己的验证模块。
> 队友拉代码后拿不到这些用例。

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| 全部用例 `points=0` | MySQL 没起 / 窗口无数据 | 查 `scada_db.scada_telemetry` 有没有数据 |
| `AttributeError: 'DiagnosisState' object has no attribute 'get'` | 用了旧的扁平字段访问 | 必须写 `state.context.xxx` / `state.data.xxx` |
| 图构建类用例 skip | 没配 `DEEPSEEK_API_KEY` | 按上面 eval 一行 |
| `ValidationError: Extra inputs are not permitted` | 往 state 里塞了未声明的键 | 对照 `src/schemas/state.py` |

---

## 2. 数据流（用测试用例 **A2** 走一遍）

A2：`PUMP-IS100-80-160-01`，`2026-09-13 07:30:00 ~ 08:09:55`（轴承劣化 → 越停机线）

| 阶段 | 谁在算 | 数据形态 | 规模 |
| :--- | :--- | :--- | :--- |
| 0 输入 | Step 1 | `context` 三件套 | 3 个值 |
| 1 取数 | Python / MySQL | `list[dict]` 11 列 | **480 帧** |
| 2 分段 | Python | 按 `operating_state` 变点切 | **480 点 → 2 段** |
| 3 统计 | Python | 每段 22 个键 | **2 × 22 = 44 个数字** |
| 4 规则判定 | Python | 机器码（段级 + 窗口级去重） | **5 种码** |
| 5 报警码 | Python | 三件套 | 4 个码 |
| 6 语义化 | **大模型** | 1~6 条现象描述 | **5 条 / 407 字** |
| 7 输出 | 图回流 | `data` 盒子 | **1 个顶层键** |

### 阶段 1｜取数（一条 SQL）

`data_tools.query_scada_telemetry(device_id, start, end)` → 每帧含 `timestamp` + 9 个测点 +
`operating_state` + `alarm_code`。**原始帧只活在函数局部变量里，不进 state。**

### 阶段 2–3｜分段与统计

```python
df['_seg_id'] = (df['operating_state'] != df['operating_state'].shift()).cumsum()
```

「**连续的同一状态 = 一段**」——同一种状态出现两次、中间隔了别的状态，就是两段。
每段算 22 个键：

| 类别 | 字段 |
| :--- | :--- |
| 段定位（4） | `start` / `end` / `duration_sec` / `data_points` |
| 水力（5） | `avg_flow` / `avg_press_out` / `avg_press_in` / `cv_flow` / `cv_press` |
| 温度（3） | `slope_temp_de` / `max_temp_de` / `max_temp_nde` |
| 振动（3） | `slope_vib_de` / `max_vib_de` / `max_vib_nde` |
| 电气（2） | `avg_motor_current` / `max_motor_current` |
| 滑窗 CV（2） | `cv_flow_peak` / `cv_press_peak` |
| 拐点（1） | `ramp_start_time`（温度斜率首次 ≥0.8 ℃/min 的时刻） |
| 判定结果（2） | `state` / `rule_hits`（本段命中的规则码） |

### 阶段 4｜规则码（机器码，不是中文句子）

`analyzer._judge_flags()` 逐段判定，把命中的**码**写进该段的 `rule_hits`，并返回窗口级去重码：

```
窗口级 threshold_flags = ['BEARING_TEMP_DE_RAMP_SLOW', 'BEARING_TEMP_DE_TRIP',
                          'BEARING_TEMP_DE_RAMP_SHARP', 'VIB_DE_TRIP', 'VIB_NDE_WARN']
段级   [DEGRADING段] = ['BEARING_TEMP_DE_RAMP_SLOW']
       [WARNING段]   = ['BEARING_TEMP_DE_TRIP', 'BEARING_TEMP_DE_RAMP_SHARP', 'VIB_DE_TRIP', 'VIB_NDE_WARN']
```

码表 `RULE_CATALOG`（**17 条**，在 `rules/thresholds.py`）给出「码 → 中文标签 + 出处」。
中文事实句由 `render_rule_hits()` 在**拼提示词时**现渲染（只进 prompt、不进 state）：

```
- [WARNING段 08:00:05~08:09:55] 驱动端温度超停机线（BEARING_TEMP_DE_TRIP）
- [DEGRADING段 07:30:00~08:00:00] 驱动端温度缓慢劣化（BEARING_TEMP_DE_RAMP_SLOW），约从 07:30:55 开始
```

### 阶段 5｜报警码三件套

从这一批帧的 `alarm_code` 列提取，**只认窗口数据、不读用户入参**：

```python
data.alarms = AlarmState(effective="TAH-101;TAHH-101;VAH-102;VAHH-102",
                         last="TAHH-101;VAHH-102",
                         all=["TAH-101", "TAHH-101", "VAH-102", "VAHH-102"])
```

### 阶段 5b｜数据质量（`quality.status` 四态）

| 情况 | status | reason |
| :--- | :--- | :--- |
| 没给时间窗口 | `EMPTY` | 未提供时间窗口：本次未做时序分析（**不查库**） |
| 窗口内没数据 | `EMPTY` | 指定时间窗口内无 SCADA 记录 |
| 数据有缺口（覆盖率 < 90%） | `PARTIAL` | 窗口内缺失约 X% 的数据（理论 N 点，实际 M 点） |
| 正常 | `OK` | 空 |

> `PARTIAL` 的口径：`理论点数 = 窗口秒数 ÷ 5 + 1`（5s 采样），实际少于理论的 90% 即判不完整。
> **下游 Step 6（安全门禁）靠它区分"看过了没毛病"和"压根没看成"。**
> 阈值 `SAMPLE_INTERVAL_SEC` / `DATA_COVERAGE_MIN_PCT` 也在 `rules/thresholds.py`。

### 阶段 6｜大模型输出（1~6 条按 type 分类）

```jsonc
data.descriptions = [
  {"type": "TREND",       "description": "驱动端温度全程单调爬升…", "evidence": ["slope_temp_de", "max_temp_de"]},
  {"type": "THRESHOLD",   "description": "驱动端温度越过停机线…",   "evidence": ["max_temp_de", "BEARING_TEMP_DE_TRIP"]},
  {"type": "CORRELATION", "description": "温度与振动同步加速上升…", "evidence": ["slope_temp_de", "slope_vib_de"]},
  {"type": "OTHER",       "description": "流量与出口压力全程平稳…", "evidence": ["avg_flow", "cv_flow_peak"]}
]
```

`type` 六选一：`TREND` / `ANOMALY` / `THRESHOLD` / `CORRELATION` / `ALARM` / `OTHER`。
**`evidence` 会按白名单过滤**（只留材料里真实出现过的指标名与规则码），编造的名字会被丢掉。

---

## 3. 关键设计决策

### 3.1 为什么只有 2 个节点？

原先 3 个节点时，取数节点必须把原始帧写进 state 才能交给下一个节点 —— 24h 窗口 **1.7 万条**
每走一步都被 LangGraph 合并一次。合并成 `analyze` 后，原始帧只是函数局部变量：

| | 3 节点版 | 现在 |
| :--- | :--- | :--- |
| 节点 | 3 | **2** |
| 私有字段 | 2（1.7 万条帧 + 报警段） | **0** |
| 每窗口查库 | 2 次 | **1 次** |
| state 里的数据量 | 4.19 MB（24h 原始帧） | **6.2 KB（压缩后结果）** |

### 3.2 为什么回写要用 `model_validate` 而不是 `model_copy`？

```python
return {"data": DataState.model_validate({**state.data.model_dump(), **updates})}
```

两个原因，都实测过：

1. **LangGraph 用返回值替换整个盒子** —— 只返回部分字段（如只给 `descriptions`），
   `analyze` 刚写好的 `metrics` / `quality` 会被一起冲掉。所以要先摊平现有内容再覆盖。
2. **`model_copy(update=...)` 不跑校验器** —— 即使开了 `validate_assignment` 也一样，
   会绕过组长 `StateModel` 里的 checkpoint 类型检查（只允许精确的
   `str/int/float/bool/None/list/dict`）。用 `model_validate` 重新构造，校验当场发生
   （有测试钉住：塞 `numpy.float64` 会被拦下）。

### 3.3 为什么"降级"用 `quality` 而不是中文告警？

告警位（`threshold_flags`）现在只放机器码。降级不是"规则命中"，塞进去会污染语义、
下游也没法用 `if` 判断。所以降级信息**只在 `data.quality` 一处**表达，且 `EMPTY` 时
**直接跳过语义化（不调大模型）** —— 没有数据就没得描述，生成一段"未做分析"的话只会污染
`descriptions`。

### 3.4 为什么阈值命中是"码进 state、句子进 prompt"？

- **码**短、稳定、机器可读：下游（Step 6 门禁、报告溯源）直接吃码，改文案不影响它们；
- **句子**人读友好，但只在拼提示词的那一刻生成，不进 state；
- 具体观测值不必重复 —— 分阶段指标里已经列了每段数值。

码表与 `_judge_flags` 的分支**一一对应**，有测试守着不许漂移（多了少了都报错）。

### 3.5 为什么 `descriptions` 是 1~6 条而不是一段话？

组长的契约把语义化产出定义成 `list[DataDescription]`：按类型分条，下游（Step 4 RAG 检索、
Step 7 报告分节）才能按需取用；一段整话既没法检索也没法筛选。
条数上限定 6 是防大模型"刷条数"把描述拆成流水账；下限定 1 是要求即使全窗口平稳也得给结论。

### 3.6 为什么判据全在 `rules/thresholds.py`？

同一物理量在不同标准里限值不同（轴承温度：GB 50275 是 80℃、IOM 文本写 85℃），
必须显式标注用了哪个；而且 **Step 2 的规则与 Step 6 的安全门禁共用同一批阈值**，
放两处必然漂移。有测试检查 `analyzer.py` 里没有规范级魔法数字。

---

## 4. 测试

| 文件 | 用例数 | 覆盖 | 需要什么 |
| :--- | ---: | :--- | :--- |
| `tests/test_nodes.py` | 30 | 契约（9 盒子）/ 子图结构 / 阈值出处 / 规则码 / 节点行为 / 回写不冲字段 / PARTIAL / 调度接线 | 无 Key 时部分 skip |
| `tests/test_pipeline.py` | 30（脚本式） | 30 个真实窗口端到端 | MySQL + Key |

**30 个端到端用例的分组**：

| 组 | 覆盖 |
| :--- | :--- |
| A 基础（4） | 纯 NORMAL / 完整故障周期 / 只截 WARNING / 只截 TRIP |
| B 故障形态（7） | 高温越线、阈值抖动、重启欠载、24h 真实密度、高碎片密度 |
| C 窗口边界（6） | 单点 / 两点 / 1 分钟 / 停机段 / 启动过渡 / **未知设备降级** |
| D 数据一致性（4） | 三个历史数据 Bug 的守护 + 跨状态边界切段 |
| E 报警码来源（5） | 只认窗口数据；入参不再被采用 |
| G 状态机（2） | 无切换 / 多状态切换 |
| H 碎片（2） | 切段粒度在碎片场景下的表现 |

**断言口径：绑物理量，不绑数据指纹。** 每条用例还会自动校验一条全局不变量：
`descriptions` 1~6 条（降级窗口要求 0 条）且 `evidence` 全部可回溯到材料。

支持的期望字段：`points` / `phases` / `alarm_count`（= 去重码种数）/ `states` /
`states_contains` / `states_set_contains` / `effective_alarm_codes(_contains/_excludes)` /
`alarm_contains` / `alarm_excludes` / `no_rule_hits_on_states` / `quality_status` /
`quality_reason_contains` / `alarm_count_min` / `max_desc_len` / `min_desc_len` / `note`。

---

## 5. 已知边界

| # | 边界 | 现状 |
| :---: | :--- | :--- |
| 1 | **分段依赖 SCADA 状态标签** | 标签标错则分段错且不报错。缓解：告警判定另有 `FLOW_ACTIVE_THRESHOLD_M3H=20` 这道只看物理量的闸门 |
| 2 | **按状态切段会产生碎片** | 阈值附近抖动会切出大量极短段（H1：7 点 4 段）；已固定为已知行为 |
| 3 | **数据集只覆盖密封泄漏一条线** | 不含气蚀 / 动不平衡 / 入口压力低场景；要恢复覆盖需重新生成数据集 |
| 4 | **CV 判据分不清"爬坡"与"高频波动"** | 重启时流量从 0 爬升会被算成波动 → 已知误报（根治需先去趋势） |
| 5 | **没有 FFT** | 数据是 5 秒一个点（0.2 Hz 采样），做频谱没有物理意义；除非将来有高频波形表 |
| 6 | **数据清洗很轻** | 只有时间戳规整 + 排序；表本身 `nullable=False`、无缺口，没什么可洗 |
| 7 | **单条报警码的时序不在 prompt 里** | 大模型只知道"窗口内出现过哪些码"，不知道每个码的起止（报警事件查询已删） |
| 8 | **没有大模型 Key 就建不起图** | `data_nodes` 导入时构造 LLM 客户端 → 相关用例只能 skip |

---

## 6. 一句话总结

> **Python 负责"算什么"，大模型只负责"怎么说"。**
>
> 480 帧原始遥测 → 2 个状态段 → 44 个统计量 → 5 种规则码 → 4 个报警码 → 5 条现象描述，
> 一路降维；大模型**碰不到原始数据**，也**写不出错数字**（它拿到的材料里没有可算的东西）。
