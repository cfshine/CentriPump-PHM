# Step 3 · vision_agent 实现说明

> **这份文档回答三件事**：
> 1. 图片从上游进来、到最终出去，**中间变成了什么形状**（用真实端到端运行 + 9 张样例图跑批逐步展示）
> 2. 每个关键设计**为什么这么做**（尤其是"为什么每条图都必须过视觉大模型""为什么 OCR 只做辅助"）
> 3. **怎么提测**（含中英文 OCR 引擎的系统级安装要求）

---

## 0. 这个模块是什么

**一句话**：把上游递来的「一批图片（路径或 URL）」，变成两份东西 —— 一份**给人读的整段描述**（给 Step 4 做 RAG 检索、给 Step 7 写报告），一份**给机器读的结构化结论**（每张图的观测 / 极性 / 读数 / 未核验项）。

```
                  ┌────────────────────────────────────────────────┐
  Step 1 / 用户 ──►│  vision_agent（Step 3）                        │
  image_refs       │  ingest ──(有图?)──► extract                   │
  (list[str])      │                        └─ 逐图 analyze_one    │
                  └────────────────┬───────────────────────────────┘
                                   │ 只回写 2 个公共字段
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
      visual_description                        visual_findings
   （→ Step 4 RAG 检索 / Step 7 报告）        （→ 机器读：极性/读数/溯源）
```

### 文件清单

| 文件 | 行数 | 职责 | 依赖 LangGraph？ | 依赖大模型？ |
| :--- | ---: | :--- | :---: | :---: |
| `vision_state.py` | 34 | 子图状态：**继承**公共契约，**无私有字段** | ❌ | ❌ |
| `vision_graph.py` | 63 | 子图定义（2 个节点 + 1 条条件边） | ✅ 唯一 | ❌ |
| `vision_nodes.py` | 372 | 2 个节点 + 单图管道 `analyze_one` + 视觉调用与重试 | ✅ | ✅ |
| `image_tools.py` | 538 | 读图 / 判 MIME / 缩放 / OCR / 判噪声 / 读数校正（全部纯工具） | ❌ | ❌ |
| `../../schemas/vision.py` | 164 | 强类型契约：`VisionFindings` / `VisionImage` / `Observation` / `Quantity` | ❌ | ❌ |
| `../../../tests/test_vision_agent.py` | 522 | 31 条单测（契约 / OCR / 校正 / 失败隔离 / 重试 / 真实图） | ❌ | 部分 ✅ |
| `../../../scripts/run_vision_samples.py` | 339 | 样例图跑批（人工验货，不进 CI） | ❌ | ✅ |

> **`image_tools.py` 不含任何 LangGraph 与大模型依赖** —— 它只吃字节、只吐文本/数值，
> 所以能脱离图和 API 单独测试。这是刻意的：**工具归工具，编排归编排，模型调用归节点**。

> **与 Step 2 的差异**：Step 2 回写 8 个公共字段（因为它要输出四份给不同下游的盒子）；
> Step 3 只回写 **2 个**（一段文本 + 一个结构化盒子），因为视觉在下游的用法只有这两种。

---

## 1. 怎么提测

### 1.1 环境准备（★ OCR 引擎是系统级依赖，pip 装不到）

```bash
cd /home/yali_ai/work_file/LLM_Project/CentriPump-PHM

# ① 解释器：项目 venv（已装 langgraph / langchain / langchain-deepseek / Pillow / pytesseract）
PY=/home/yali_ai/work_file/.venv/bin/python

# ② ★ OCR 引擎（Tesseract）—— 必须用系统包管理器装，pip 只能装"调用它的胶水"
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng

#    验证（应能看到 chi_sim 与 eng 两个字库）：
tesseract --version            # → tesseract 4.1.1
tesseract --list-langs         # → chi_sim / eng / osd

# ③ Python 侧的胶水 + 图像库
$PY -m pip install pytesseract Pillow

# ④ 大模型凭据：走系统环境变量（代码里不做任何显式声明）
eval "$(grep -E '^export DEEPSEEK_API_KEY=' ~/.bashrc)"

# ⑤ （只在跑"主图端到端"时才需要）MySQL 127.0.0.1:3306 / scada_db，配置在项目根 .env
```

**为什么两步都要做**：`pytesseract.image_to_string()` 本质是「把图片写成临时文件 →
执行一次 `tesseract 图片 stdout -l chi_sim+eng` → 读回 stdout」。
`apt` 装的是**引擎 + 中英文字库**，`pip` 装的是**Python 调用它的包装器**，缺一不可。

> 中文场景必须装 `tesseract-ocr-chi-sim`，只用 `eng` 的话中文界面/铭牌会大面积漏认。

**依赖缺失会怎样**（都不会让程序崩，但要清楚代价）：

| 缺什么 | 现象 | 影响 |
| :--- | :--- | :--- |
| 缺 `tesseract` 可执行文件 / 缺 `chi_sim` | `run_ocr()` 返回 `ok=False` + 原因 | **不崩**：如实写进 `limitations`（"OCR 未生效（…）：本图未经文字提取"），图照常送模型；代价是**丢了文字辅助 + 没法用 OCR 校正读数** |
| 缺 `pytesseract` | 同上（`reason="未安装 pytesseract"`） | 同上 |
| 缺 `Pillow` | `ModuleNotFoundError` 直接抛出 | 该图变成 `basis="failed"` 占位（可溯源），整批不崩 |
| 缺 `DEEPSEEK_API_KEY` | `init_chat_model` 在 import 期就报 ValidationError | 子图根本建不起来（测试会明确 skip 并说明原因） |

### 1.2 三条提测命令

```bash
# ① 契约 + 工具 + 失败隔离（不需要网络；有 Key 时会跑 1 张真图）
$PY -m pytest tests/test_vision_agent.py -q
#    期望：31 passed
#    覆盖：状态继承 / 公共契约形状 / MIME / data URL / OCR 判噪声 /
#          读数校正四类 / 失败隔离三条 / 重试两条 / 真实图 live

# ② Step2 + Step3 一起跑（契约与挂载）
$PY -m pytest tests/test_nodes.py -q
#    期望：9 passed   ← 与上面合计 40 passed

# ③ 样例图跑批（**真实调 API、要花钱**，人工看识别质量）
$PY scripts/run_vision_samples.py --json /tmp/vision_report.json
#    期望：汇总：成功 9 / 失败 0
#    可用参数：--dir / --limit N / --fail-fast / --json
```

### 1.3 命令失败自查

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| 跑批里每张图都 `OCR 体检 : 噪声→丢弃 | 0 字` | Tesseract 没装好 / 缺字库 | `tesseract --list-langs`；按 §1.1 ② 重装 |
| `ModuleNotFoundError: No module named 'PIL'` | 解释器选错了（旧 venv 没装 Pillow） | 用 `/home/yali_ai/work_file/.venv/bin/python` |
| 测试全部 `SKIPPED: 需要 DEEPSEEK_API_KEY` | 当前 shell 没导出 Key | `eval "$(grep -E '^export DEEPSEEK_API_KEY=' ~/.bashrc)"` |
| `RuntimeError: 视觉结构化输出解析失败（返回 None）` | 输出被 `max_tokens` 截断（连重试 3 次都没成） | 提高 `llm_client.py` 的 `max_tokens` |
| `basis=failed` + `FileNotFoundError` 占位 | 图片路径写错 / 文件被移走 | 看占位结论里的 `source_ref` 与失败原因 |
| 跑批很慢（每图 7~15s） | 正常：一次 OCR + 一次视觉 API | 见 §2 耗时分解；批内并行尚未实现（§6-1） |
| 挂载测试 skip | 没配 Key | 同第 3 行 |

> `scripts/run_vision_samples.py` 是**人工验货工具**，不是 pytest 用例：
> 它对每张图单独 try/except（生产代码没有这层"体贴"），失败只打印不断言，退出码 1 表示有图失败。

---

## 2. 完整数据流

### 📊 全局一览：一次真实端到端（主图 `START → step2 → step3 → END`）

窗口：`PUMP-IS100-80-160-01`，`2026-09-13 10:25:00 ~ 10:55:00`（**FAL-104 真实发生时段，10:30 起**），
图片：`leak_close.png`（物理泄漏照）+ `hmi_alarm.png`（HMI 告警截图）

| 阶段 | 谁在算 | 数据形态 | 实测耗时 |
| :--- | :--- | :--- | :--- |
| 0 输入 | Step 1 / 用户 | 5 个 A 类字段 | — |
| 1 Step 2 | Python + MySQL + 大模型 | 8 个 B 类字段（时序结论） | 数秒（含查库与语义化） |
| 2 Step 3 | 2 个节点 + 2 个函数链 | **2 个 C 类字段** | **7.4 s（图1）+ 7.4 s（图2）** |
| 3 输出 | 图回流 | 主图共 **15 个公共字段** | — |

---

### 阶段 0 ｜输入：Step 1 / 用户写进父图 `DiagnosisState`

```json
{
  "device_id": "PUMP-IS100-80-160-01",
  "start_time": "2026-09-13 10:25:00",
  "end_time":   "2026-09-13 10:55:00",
  "alarm_code": "FAL-104",
  "image_refs": ["data/sample_inputs/vision/leak_close.png",
                 "data/sample_inputs/vision/hmi_alarm.png"]
}
```

Step 3 只需要其中 **2 个**：`image_refs`（要处理什么）与 `device_id` / `alarm_code`（填提示词）。
**窗口、阈值、报警码的判定全都不归它管。**

---

### 阶段 1 ｜Node 1 `ingest_node`：清洗清单 + 空图短路

```python
refs = [str(r).strip() for r in state["image_refs"] if str(r).strip()]
```

- **没图** → 回写 `{"image_refs": [], "visual_description": "", "visual_findings": VisionFindings()}`，
  条件边直接走 `END`：**不读图、不 OCR、不调模型，成本为零**
- **有图** → 回写清洗后的清单，进入 `extract_node`

> 这一步**一个函数都不调**，实测 ~0 ms。

---

### 阶段 2 ｜Node 2 `extract_node`：逐图跑 `analyze_one`

以 `hmi_alarm.png` 为例（实测 7.42 s）：

| # | 调用 | 真实输入 → 输出 | 耗时 |
| :---: | :--- | :--- | ---: |
| 1 | `read_image_bytes` | 路径 → **1.27 MB 字节** | 1 ms |
| 2 | `detect_content_type` | 文件头 `\x89PNG` → `"image/png"` | 0 ms |
| 3 | `run_ocr`（**在原图上**） | 图 → `"\| \|  Pump Monitoring System…（392 字）"` | **1.9 s** |
| 4 | `assess_ocr` | 392 字 / 81 数字 / 命中单位×1、字段名×5 → **`credible=True`** | 0 ms |
| 5 | `resize_for_api` | 1.27 MB / 1672×941 **已在限制内 → 原样直传** | **0 ms** |
| 6 | `_vision_understand` | 图 + OCR 文本 → 11 条观测 + 描述 + 4 条未核验 | **5.5 s** |
| 7 | `apply_ocr_readings` | 校正读数（本次命中 0 条，只做一次遍历） | 0 ms |
| 8 | `VisionImage(...)` | 组装并 Pydantic 校验 | 0 ms |

**第 3 步的真实输出**（Tesseract 原文，节选）：

```
| |  Pump Monitoring System
Centrifugal Pump Group
四           Trend          |          Alarms        Reports      Settings
12.5 WL    8.9 m?/h
0.25  MPa
78
Ga Pump P-101  | ”Model 1IS100-80-160  Location Pump Room 1   Status RUN
© 2026-09-13 00:41:07
```

> **看出来了吗**：`IS100-80-160` 被认成 `1IS100-80-160`、`m³/h` 变成 `WL`/`m?/h`、
> 而且**整条 `ALARM FAL-104 LOW FLOW` 告警横幅根本没被认出来**。
> 这就是"OCR 只做辅助、每条图必须过模型"的实测依据（详见 §3.6）。

**第 6 步的真实输出**（模型给的结构化结论，节选）：

```jsonc
{ "image_kind": "hmi_or_screenshot", "basis": "vision+ocr", "confidence": 0.9,
  "natural_description": "这是一张离心泵监控系统（Pump Monitoring System）的 HMI 截图，
    顶部为红色告警条，显示 ALARM FAL-104 LOW FLOW，时间戳 2026-09-13 00:41:07…",
  "limitations": ["截图为 HMI 界面，非现场实拍，无法看到泵体、管路、密封等物理外观",
                  "趋势图曲线为像素级目测读数，未标注数据点",
                  "OCR 摘录中部分字符存在识别错误，已按画面实际显示校正"],
  "observations": [
    {"target":"顶部告警条","finding":"红色告警条显示 ALARM FAL-104 LOW FLOW…",
     "polarity":"abnormal","quantity":null,"evidence":"ALARM FAL-104 LOW FLOW"},
    {"target":"Flow 卡片 PV 过程值","finding":"显示 PV 8.9 m³/h，低于同卡片 SP 12.5 m³/h",
     "polarity":"abnormal","quantity":{"name":"flow_pv","value":8.9,"unit":"m³/h"}}]}
```

### 阶段 3 ｜输出：只回写 2 个公共字段

```
子图 state（15 个键）= 公共契约 15 个：
  [A 类·输入] device_id / start_time / end_time / alarm_code / image_refs
  [B 类·Step2] calculated_metrics / threshold_flags / effective_alarm_codes /
               last_alarm_codes / all_alarm_codes_in_window /
               llm_description / basic_judgment / rag_search_queries
  [C 类·Step3] visual_description   ← 人读（Step4 RAG 输入 / Step7 报告）
               visual_findings      ← 机器读（VisionFindings 强类型盒子）

私有字段：无（Step 3 一个都没有）
```

**`visual_description` 的真实内容**（由 `visual_findings` 派生，唯一写入者）：

```
[图1] leak_close.png
画面为卧式离心泵组（灰黑色泵体 + 蓝色电机与轴承箱）的近景。泵体与轴承箱之间的填料压盖 /
密封压盖区域可见明显的深色油性液体，沿压盖下缘连续向下流淌，在泵体下方的底座与基础面上
形成一滩积液，液面反光。压盖及其周边螺栓、泵体下部表面有深色油污与锈迹附着。
本图未核验：无 OCR 结果，画面中未见铭牌、表计读数等文字信息，无法提供任何数值读数；
泄漏液体的具体性质（油/水/介质）无法从画面判断；泄漏起始时间与泄漏速率无法从单张静态图像判断

[图2] hmi_alarm.png
画面为离心泵监控系统（Pump Monitoring System / Centrifugal Pump Group）的 HMI 截图…
本图未核验：趋势图纵轴刻度与曲线为像素级读取…；OCR 摘录中部分字符（如 'OW)'、'NP'）为识别噪声
```

### 📊 这一步的价值：两个模态对上了

```
Step2 时序：effective_alarm_codes = "FAL-104;IAL-105"
            threshold_flags = ["流量工况偏离额定值 >15% (实际 -58.2%)",
                               "2分钟滑窗内流量/压力高频波动 (CV_flow_peak=82.3%)",
                               "电机电流欠载 (avg 9.43 A)"]
Step3 视觉：HMI 截图读到「ALARM FAL-104 LOW FLOW」
            泄漏照片读到 2 条「深色液体沿压盖流淌 + 底座积液」的 abnormal 观测
```

---

### 📊 9 张样例图跑批（`scripts/run_vision_samples.py`）

| 图片 | basis | image_kind | confidence | abnormal | 说明 |
| :--- | :--- | :--- | ---: | ---: | :--- |
| `leak_close.png` | vision | leak_or_stain | 0.82 | 3 | 密封压盖渗漏（核心用例） |
| `leak_wide.png` | vision | leak_or_stain | 0.62 | 2 | 泵组全景 + 地面湿渍 |
| `gauge_low.png` | vision | gauge_or_meter | 0.82 | 0~3 | 压力表读数（模型给 `quantity`） |
| `hmi_alarm.png` | **vision+ocr** | hmi_or_screenshot | 0.90 | 3 | 唯一 OCR 有用的一张 |
| `nameplate.png` | **vision+ocr** | nameplate | 0.90 | 0 | 铭牌参数全对（OCR 只抓到部分） |
| `thermal_bearing.png` | vision | thermal | 0.72 | 0 | 热成像 |
| `bearing_rust.png` | vision | mechanical_surface | 0.62 | 4 | 锈蚀/渗油 |
| `bad_quality.png` | vision | scene | **0.25** | 0 | 逆光+模糊：置信度最低、未核验项最多 |
| `normal_control.png` | vision | scene | 0.62 | **0** | 正常对照组：**没有误报异常** |

> `confidence` / `abnormal` 条数在多次运行间有 ±0.1 级波动（模型侧固有波动，`temperature=0` 也不能完全消除）。

### 📊 单图耗时分解（优化后实测）

| 环节 | 占用 | 备注 |
| :--- | ---: | :--- |
| `_vision_understand`（视觉 API） | **74~84%** | 模型侧，7~13 s 量级 |
| `run_ocr`（1 遍 Tesseract） | **16~26%** | 0.8~1.9 s，与图大小/文字量相关 |
| 其余全部（读盘 / 判型 / 缩放 / base64 / 校正 / 拼文本） | **< 0.1%** | 合计 <5 ms |

---

## 3. 关键设计决策与理由

### 3.1 为什么是 **2 个节点**，而不是 4 个或 1 个？

草案里原本画的是 4 个节点（`ingest → route → ocr/vision → normalize`）。最终合并成 2 个：

| 方案 | 问题 |
| :--- | :--- |
| 4 节点 + 条件边 | `route` 要先"猜"图属于哪类再选分支；而**猜错的代价比多跑一次 OCR 大得多**（颜色数猜类型已被实测证伪） |
| 1 个节点 | 空图短路与"清洗清单"就会和图处理混在一处，可读性下降 |

现在：**`ingest` 只做清单清洗与空图短路，`extract` 里用普通 for 循环逐图处理**。
多图不靠 LangGraph 的循环边（那会带来状态合并的复杂度），算法全在 `analyze_one` 这个纯函数里 —— 与 Step 2「算法在函数里、图只编排」同一风格。

### 3.2 为什么图片字节**不进 State**？

两个节点之间只传 `image_refs`（字符串清单），字节是 `analyze_one` 内部的局部变量。

**理由**：LangGraph 的 State 每经过一个节点都要**整份合并一次**。几百 KB ~ 几 MB 的图片字节进去，
每次状态合并都在拷贝它。而它唯一的用途就是"喂给 API 一次"，用完即弃。

**副产品**：State 里没有任何 bytes 字段 → 契约干净、序列化/SQLite checkpointer 都不会爆。

### 3.3 为什么 OCR 跑在**缩放之前**？

```python
original = read_image_bytes(ref)
ocr = run_ocr(original)          # ① 先在最高清像素上认字
...
data = resize_for_api(original)  # ② 只有要发 API 才缩放
```

字越小越怕缩放：一张 4000px 的 HMI 截图若先缩到 2048 再 OCR，小字识别率会明显下降。
缩放是"为了迁就 API 的体积/边长限制"，与"认字"是两件事，顺序不能颠倒。

### 3.4 为什么 OCR 的判定只有"**噪声 / 有用信息**"两个结论？

```
credible = 字符数 ≥ 24  且  数字个数 ≥ 4  且  (命中单位 / 位号 / 字段名 之一)
```

三条判据的用意：

| 判据 | 挡掉什么 |
| :--- | :--- |
| 字符数 ≥ 24 | 零散噪声（真实照片常吐 `"ww Wink"` 这种 7 个字符的东西） |
| 数字 ≥ 4 | 纯字母/纯符号噪声（我们要的是读数与位号） |
| 单位/位号/字段名 | 证明这段文字**属于工业语境**，不是随机纹理 |

**结论只有一个布尔值**：`True` → 塞进提示词当参考 + 用它校正读数；`False` → 整段丢弃。

> 曾经还有第二个结论 `text_dominant`（"够完整到可以跳过视觉模型"），**已按用户口径整体删除**（§3.6）。

### 3.5 为什么行数、Tesseract 置信度**不参与判断**？

它们曾经参与打分，后来实测发现会骗人：**热成像刻度数字的 Tesseract 平均置信度高达 94**（比理想文字屏的 77 还高），
可整张图只有 22 个字 —— 只看置信度会把"一堆零散数字"当成有用信息。

现在这两个数**已从代码里整体删除**（连字段都没了），因为它们的另一个用途（诊断打印）也不值得再跑一遍 Tesseract（见 §3.8）。

### 3.6 ★ 为什么**每条图都必须过视觉大模型**（不省钱）？

这是本模块最重要的一条决策，且有实测依据：

| 信息 | Tesseract（OCR） | 视觉大模型 |
| :--- | :--- | :--- |
| **关键告警码 `FAL-104`** | ❌ **完全没认出来** | ✅ 认出来了 |
| 型号 `IS100-80-160` | ❌ 认成 `1S100-80-160` | ✅ 正确 |
| 单位 `m³/h` | ❌ 认成 `m?/h`、`WL` | ✅ 正确 |
| 铭牌 `100 m³/h` / `32 m` | ❌ **漏掉**（只抓到 `2900 r/min`、`15 kW`） | ✅ 全对 |
| 平均置信度 | 68（低于"可信"经验线） | — |
| 速度 | **快**（1~2 s） | 慢（7~13 s + 花钱） |

**结论**：OCR 的价值在"快"和"当交叉核对"，**不在"准"**。
省下那一次模型调用，代价是把 `FAL-104` 这种诊断起点弄丢 —— 不值。

所以旧的"**OCR 独证**（文字够完整就不调模型）"路径已整体删除：
`Basis` 枚举从 `vision / vision+ocr / ocr / failed` 收缩为 **`vision / vision+ocr / failed`**，
`_image_from_ocr()`、`text_dominant`、`TEXT_DOMINANT_SCORE`、`OCR_HIGH_CONF`、`guess_text_kind()`、六项打分全部删除。

### 3.7 为什么 `resize_for_api` 要有一条"**已在限制内就原样返回**"的快速通道？

**实测**：9 张样例图 **9/9 都在限制内**（≤2048px、≤8MB），却每张都做了"解码 → `PNG optimize=True` 重新编码"，
**耗时 0.7~4.6 秒，占单图总时间的 32%** —— 纯白工。

```python
with Image.open(io.BytesIO(data)) as probe:   # 只读文件头，不解码像素（微秒级）
    width, height = probe.size
if len(data) <= max_bytes and max(width, height) <= max_side \
        and content_type in {"image/jpeg", "image/png"}:
    return data                                # ← 原字节返回
```

顺带一个质量红利：原样返回**保住了原图的 alpha 通道**（旧的 `_open_rgb` 会把透明压成白底）。
GIF/WebP 刻意不走快速通道（它们原本会被转 JPEG 并取第一帧，保持老行为不变）。

### 3.8 为什么删掉"第二遍 Tesseract"？

`assess_ocr` 曾经为了拿"行数 + 平均置信度"再调一次 `pytesseract.image_to_data` ——
**同一个引擎、同一张图、跑第二遍**，实测 **0~1.7 秒/图（占 12%）**，而那两个数（§3.5）**不参与任何判断**。

现在：`_text_layout()` 整个函数删除，`OcrAssessment` 去掉 `lines` / `confidence` 两个字段。
真要看这两个数，跑批脚本可以按需自己量（生产链路不需要）。

### 3.9 为什么**禁止清洗 OCR 文本**？

有人提议"用正则把 OCR 文本里的多余空格/噪声洗掉"。**结论是不洗**，理由三条：

1. **收益只有 token**：实测清洗省 24% 字符，但**对"噪声/有用"判断零影响**
   （判据本来就把空白剥掉了：`len(re.sub(r"\s+","",text))`）。
2. **字符串里的"排版"多半是误识别**：HMI 那张的 `| |  Pump`、`四` 是 Tesseract 把界面边框/图标认成了字符
   （逐词置信度 24/44/12，而真文字都是 90+）。保留它们不是"保住结构"，是把噪声当信号。
3. **风险不对称**：清洗规则一旦存在，很容易被"顺手加强"，而某张图上那恰好是内容。
   这个项目里"数据不能错"的优先级高于"省几个 token"。

> 真正的排版信息在 `image_to_data` 的 **x/y 坐标**里（`block/paragraph/line` + `left/top`），
> 不在字符串里。将来若真需要还原版面结构，应该用坐标，而不是猜空格。

### 3.10 为什么读数校正要设计成"**只在紧邻位置取数**"？

`apply_ocr_readings` 的用途是"**数字以 OCR 为准**"（模型"看"数字不如传统 OCR 认字准）。规则：

```
1. 拿 quantity.name（如"出口压力"）在某 OCR 行里找位置
2. 名字**紧后面**（允许空白/冒号/等号）必须就是一个数字，否则完全不动
3. 单位兼容才覆盖 value；不兼容则不改数、只记"未采信"
4. 覆盖时在 evidence 写一行 "OCR 校正：<该行原文>"；不覆盖也留痕
5. 返回**深拷贝**，不改调用方传入的对象
```

**旧实现的三个坑（都已堵）**：

| 坑 | 后果 | 现在 |
| :--- | :--- | :--- |
| 正则写成 `[-+]?\d+` 并扫"整行第一个数字" | OCR 行 `出口压力 PI-102 0.42 MPa` 抠出 **-102.0**，覆盖掉正确读数 | 数字必须**紧跟在读数名之后**，且不允许紧贴字母/连字符 → 位号里的 `102` 天然出局 |
| 静默改数、无痕迹 | 无法审计、无法发现 bug | 改与不改都写进 `evidence` |
| 原地修改传入对象 | 复用对象时被污染（测试踩过） | 深拷贝 |

> **实测触发率是 0**：9 张样例图 + 全部跑批里，从未触发过校正（真实模型几乎总把读数写进描述，
> 且 `quantity.name` 后面紧跟数字的排版很少见）。它的耗时是 **0 ms**，留着成本为零。

### 3.11 为什么一张图失败不能拖垮整批？

**实测过的旧行为**：`extract_node` 里没有逐图兜底，某张图抛异常会**冲出整个函数** ——
循环里那个局部列表 `images` 随栈展开消失，**前面已经成功（而且已经花掉模型调用）的图，结论一个字都留不下**；
LangGraph 还会把整个 super-step 判为失败，主链路整体抛异常，Step 3 之后全空。

**现在**：每张图独立 `try/except`，失败产出一条 **占位结论**：

```jsonc
{ "source_ref": "…/坏图.png", "basis": "failed", "confidence": 0.0, "observations": [],
  "natural_description": "（本图识别失败，未产生视觉结论：FileNotFoundError: …）",
  "limitations": ["处理失败：FileNotFoundError: …"] }
```

**为什么是占位而不是"跳过这张"**：批量识别时其他图的结论是真金白银换来的；
而且失败必须**可溯源** —— 读者要能看出"第 N 张没识别成功、原因是什么"，而不是只发现"少了一张"。
下游代码一句 `basis == "failed"` 就能挑出这些占位。

### 3.12 为什么重试只有**一层**？

视觉模型失败通常是网络波动或余额不足。原来的实际行为是**双层重试**：
底层 OpenAI SDK 默认 `max_retries=2`，我们外面再 3 轮 → 最坏 **3 × (1+2) = 9 次真实 HTTP 请求**
（每次都是完整图文请求、会重复计费），而日志里只看到"第 3 次失败"，真实次数不可见。

**现在**：`vision_model` 显式 `max_retries=0`，重试权收归 `_vision_understand` 一层：

| 情况 | 行为 |
| :--- | :--- |
| 网络波动 / 连接重置 / 超时 / 限流 / 5xx | **重试**，最多 3 次（含首次），退避 **1s → 2s** |
| 结构化输出返回 `None`（多为截断） | 也重试；用尽才报错 |
| **余额不足**（`Insufficient Balance` / `402`） | **立刻抛**，不重试、不等待 |
| 鉴权失败 / 模型名写错 / 请求不合法 | 立刻抛 |

**判定方式用"黑名单"而不是"白名单"**：白名单（只重试超时/连接错误）会漏掉没预料到的瞬时故障；
黑名单只列"确定重试无用"的几类，宁可多等几秒，也别把一次本来能成功的识别判死。

> 同文件里还设了 `timeout=60`：SDK 默认读超时是 **600 秒**，一次"连上但不回数据"就能卡 10 分钟，
> 3 轮重试会变成最坏 30 分钟。正常调用是 7~13 秒，60 秒足够暴露异常。
>
> ⚠ 文本模型 `model`（Step 2 用）**保持不动**：Step 2 没有自己的重试层，关掉 SDK 重试会让它失去保护。

### 3.13 为什么公共契约只有 **2 个字段**？

| 字段 | 消费者 | 形态 |
| :--- | :--- | :--- |
| `visual_description` | **人**：Step 4 拿来生成 RAG 检索词、Step 7 写报告 | 一段文本（每图一段 + `[图N]` 锚点 + `本图未核验：…`） |
| `visual_findings` | **代码**：遍历/计数/过滤/比对 | `VisionFindings` 强类型（不是 `dict[str, Any]` 袋子） |

**为什么必须是两份而不是一份**：

- 只有文本 → 下游无法可靠地机器处理（同一意思十几种写法；**"未见泄漏"也含"泄漏"两个字**，字符串匹配会误判）
- 只有结构 → "给报告的人话"缺失，每个消费者自己去拼文本必然分裂

**关键规则：文本只能由盒子派生，且只有 `_compose_description` 一个写入者。**
只要写入者唯一，两份表示就不是"两处真相"，而是"同一处真相的两种视图"，永不漂移（有测试守护）。

**被剔除的字段**（都曾存在过）：`content_type`（纯技术细节）、`extracted_text`（原文归 `evidence` 与观测）、
`quantity.raw`（与 `evidence` 重复）、`rag_queries` / `visual_rag_queries`（Step 4 自己是 LLM 节点，从描述现推更准）。

### 3.14 为什么子图**没有私有字段**？

原来有 6 个（`route` / `image_kind` / `ocr_text` / `content_type` / `current_ref` / `need_vision`），
但它们只记录"**最后一张图**"的值（多图时语义就是误导性的），且没有任何消费者。

现在：中间量（OCR 体检、basis、置信度）只活在 `analyze_one` 的返回值里，**不进 State**；
图只回写 2 个公共字段；逐图诊断改成节点内一行日志。
`VisionAgentState` 仍保留自己的类（对称 Step 2、将来要加私有字段也有落点）。

### 3.15 为什么 `limitations`（"没看清什么"）**保持现状**？

三个来源，性质完全不同：

| 来源 | 依据 | 可审计？ |
| :--- | :--- | :---: |
| **主来源：大模型自己写的**（提示词要求"看不清/反光/被遮挡的内容写入 limitations，不要编造"） | 模型看图后的**主观申报** | ❌ |
| 代码追加：`run_ocr` 返回 `ok=False` 时写"OCR 未生效（原因）" | 系统事实 | ✅ |
| 代码追加：整图处理失败时写"处理失败：<异常类型>: <消息>" | 异常对象 | ✅ |

**已知边界**（都不打算现在改）：没有条数校验与上限；**条数不是质量信号**（实测 confidence 从 0.25 到 0.92，
条数恒在 4~6，因为它是按提示词"写齐"的）；是自由文本、无分类，代码无法区分"遮挡/分辨率/角度/参数缺失"。

**改成机器信号（枚举分类 / 加 `[系统]` `[模型]` 前缀）是为想象中的需求付成本** ——
等 Step 7 真的要拿它填工单的"未覆盖范围"时，按那个栏位需要的格式再设计。

### 3.16 为什么提示词里**不写 JSON 格式**？

因为用了 `with_structured_output(VisionLLMOutput)` —— **schema 由 LangChain 自动注入**。
提示词里再手写一遍既冗余又容易和 schema 冲突。

`VisionLLMOutput`（模型填的）与 `VisionImage`（我们造的）**刻意分开**：
`source_ref` / `basis` 是**我们**知道的（哪张图、走没走 OCR），不该让模型编。
`VisionLLMOutput` **不加** `extra="forbid"`（模型多吐一个键时宁可忽略，也别让整条输出校验失败 =
这次调用白花钱）；而 `VisionImage` / `VisionFindings` 加 `forbid`（我们构造，字段名写错要当场报错）。

### 3.17 为什么 `confidence` 的来源只有两处？

```python
confidence = 模型自评（0~1）  或  0.0（处理失败占位）
```

**没有"写死的中低值"了**：OCR 独证路径删除后，那个"固定 0.4（表示无视觉核验）"的特殊值随之消失。
现在 `confidence` 的语义是单一的："这次识别有多可信"，下游排序/卡点不必再查"这个数是哪来的"。

---

## 4. 测试用例清单（31 条）

跑法：`$PY -m pytest tests/test_vision_agent.py -q` → 期望 `31 passed`
（连同 `tests/test_nodes.py` 的 9 条，合计 **40 passed**）

### 分组与意图

| 组 | 条数 | 覆盖意图 |
| :--- | ---: | :--- |
| **契约与状态** | 2 | 子图状态继承且**无私有字段**；公共契约只共享"描述 + 强类型盒子" |
| **工具（无 API）** | 5 | MIME 判型（PNG/JPEG/垃圾字节）；data URL 前缀；`resize_for_api` 快速通道与超限仍缩 |
| **OCR 判噪声/有用** | 5 | 理想文字屏有用；真实 HMI 有用；`"ww Wink"` 是噪声；空文本不调 Tesseract；无单位/位号/字段名 → 仍算噪声 |
| **OCR 结果语义** | 1 | `OcrResult` 区分"图上没字"与"OCR 没跑成" |
| **读数校正** | 4 | `-102` 回归；紧邻取数；单位冲突不覆盖；不改入参 |
| **OCR 失败上报** | 1 | OCR 挂掉仍走模型 + 如实写进 limitations（并出现在文本里） |
| **重试** | 3 | 单层（两处 `max_retries==0`）；网络波动重试成功（3 次）；余额不足不重试不等待 |
| **文本派生** | 1 | `_compose_description` 形态：锚点 / limitations 折叠 / 无 limitations 不出那行 |
| **图编排** | 4 | 空图不调模型；Step3 挂在主图；有用文本进提示词但**绝不跳过模型**；父图只见契约字段 |
| **失败隔离** | 3 | 一张坏图不影响好图；失败可溯源（盒子 + 文本）；全坏时形状仍恒定 |
| **真实图 live** | 2 | 样例图存在性；跑通一张（锚点 + 盒子 + `natural_description` 必须出现在文本里） |

### 三个值得单独说的用例

**① `test_batch_keeps_good_images_when_one_fails` —— 失败隔离回归**
```
输入 3 张：["good1.png", "坏图.png", "good2.png"]
期望：结论条数 == 3，basis == ["vision", "failed", "vision"]
      且两张好图的描述都还在文本里
```
这条守护的是"**一张坏图不许让整批一起失败**"（旧版会把已成功的结论全丢掉，§3.11）。

**② `test_ocr_correction_never_turns_tag_number_into_negative` —— `-102` 回归**
```
输入 OCR 行："出口压力 PI-102 0.42 MPa"（模型 value=0.5）
期望：value 仍是 0.5（旧版会覆盖成 -102.0）
```

**③ `test_useful_ocr_text_is_passed_to_model_but_never_skips_it` —— "辅助而非替代"**
```
monkeypatch _vision_understand 记录收到的 ocr_text
期望：basis == "vision+ocr"；描述来自模型；提示词里含 "ALARM FAL-104"；confidence == 模型自评
```

---

## 5. 怎么新增/修改测试用例

测试集中在 `tests/test_vision_agent.py`，分两类：

| 类型 | 需要什么 | 写法 |
| :--- | :--- | :--- |
| **纯函数/工具测试** | 不需要 Key、不需要网络 | 直接调 `evaluate_ocr_quality(...)` / `apply_ocr_readings(...)` / `resize_for_api(...)` |
| **需要建图的测试** | 需要 `DEEPSEEK_API_KEY`（`vision_nodes` 在 import 期就构造模型） | 加 `@requires_key`；用 `monkeypatch` 换掉 `run_ocr` / `_vision_understand`，**不打真 API** |
| **真实图 live** | 需要 Key + `data/sample_inputs/` 下有图 | 只挑 1 张（批量请用跑批脚本，别塞进 pytest —— 那是真花钱的） |

**约定（重要）**：

1. **测试替身不写进生产代码** —— 假模型/假 OCR 一律用 `monkeypatch` 在测试侧注入。
2. **不要再用"吞掉异常 → skip"的写法**：只有两个前置条件缺失才允许 skip
   （没 Key / 没样例图），**图建不起来、调用失败都必须红**。历史上这种写法让整个 Step 3 主链路
   "零覆盖还显示绿色"。
3. **断言口径：绑语义，不绑数据指纹。**
   ✅ `basis == "vision+ocr"`、`"OCR 未生效" in limitations`、`value == 0.5`（回归用）
   ❌ 绑死某次运行恰好有多少条观测、confidence 恰好是 0.9（模型侧有固有波动）

### 加用例的步骤

1. **先跑一次拿真实值**（别猜）：
   ```bash
   $PY scripts/run_vision_samples.py --dir data/sample_inputs/vision --limit 2
   ```
   或对纯函数直接构造入参。
2. 在 `tests/test_vision_agent.py` 里加用例，**docstring 写清"这条守护什么"**（推荐写明历史 bug）。
3. 跑 `$PY -m pytest tests/test_vision_agent.py -q`，确认全绿。

---

## 6. 已知边界

| # | 边界 | 影响 | 现状 |
| :---: | :--- | :--- | :--- |
| 1 | **多图串行** | 9 张图 ≈ 9 × 7 s ≈ 60 s（每次调用彼此独立、且是 I/O 等待） | 未做并行。`configs/config.yaml` 已有 `concurrency.max_workers: 4`，用线程池预计可降到 **~15 s**。要做需保证：结果顺序与输入一致、失败隔离仍生效、别把限流打爆 |
| 2 | **OCR 对真实截图不够准** | 漏认 `FAL-104`、`IS100`→`1S100`、`m³/h`→`m?/h`；铭牌漏掉 `100 m³/h` 与 `32 m` | 这是"每图必过模型"的直接原因（§3.6）。缓解：模型会自行按画面校正，并写进 `limitations` |
| 3 | **OCR 判噪声是"整段一个结论"** | 400 字里 95% 垃圾 + 5% 真内容 → 判"有用"，垃圾一起进提示词（HMI 实测就是这样） | 未做逐词过滤。方向：`image_to_data` 的**逐词置信度**（实测垃圾词 12~44、真文字 90+）比正则猜精确，但当前判错两个方向都不伤结论，收益低 |
| 4 | **`apply_ocr_readings` 实测触发 0 次** | 纠错能力"在册但没生效过" | 保留（0 ms、逻辑已修好并有用例守护）。若要求减代码可删，**但删了不提速** |
| 5 | **`limitations` 是自由文本、无校验、无上限** | 无法用代码区分"遮挡/分辨率/角度/参数缺失"；条数与"问题多少"无关 | 按用户口径保持现状（§3.15）。有消费者时再上枚举 |
| 6 | **演示图的时间戳是编的** | `hmi_alarm.png` 画面写 `00:41:07`，而库里 FAL-104 真实发生在 `10:30` —— 跨模态时间对不上 | **生成图片时请把时间戳改成真实窗口**；跨模态时间对齐本身是 Step 5 的事 |
| 7 | **`configs/config.yaml` 的 `vision:` 段是死配置** | 没人读它（模型参数都在 `src/utils/llm_client.py`） | 待收敛：模型参数统一到 config 或统一到 client |
| 8 | **`extract_node` 里单张图的成本不可再见** | 逐图 `basis/置信度/异常数` 只在节点日志里，不进 State | 这是契约瘦身的代价；跑批脚本会打印，需要结构化诊断时看日志 |
| 9 | **文本模型（Step 2）没有自己的重试层** | 它依赖 SDK 默认 `max_retries=2` | 刻意不动（属 Step 2 改动范围）。将来 Step 2 要做自己的重试，按 §3.12 同样收敛 |
| 10 | **未见过的图类可能被误判** | `image_kind` 是模型填的枚举，新设备/新场景可能落到 `other` 或选错类 | 枚举已封闭（8 类），下游按 `basis` / `polarity` 而不是 `image_kind` 做关键判断更安全 |

---

## 7. 给编排方（Step 1 / 组长）的对接说明

> **本节是写给"决定主图怎么排"的人看的**：不需要读代码，看下面三张表就能拼。
> Step 2 与 Step 3 **互相独立**（谁都不读对方的产出），所以它们可以**串行、并行、或各自跳过**。

### 7.1 两个子图需要什么、给出什么

| 子图 | 必须有 | 可选 | 缺"必须有"时的行为 |
| :--- | :--- | :--- | :--- |
| **Step 2** `data_agent` | `device_id` + `start_time` + `end_time` | `alarm_code` | ✅ **软降级**（2026-09-17 起）：不查库、不计算、不调大模型，返回结构完整的空结果 + 告警 `未提供时间窗口：本次未做时序分析` |
| **Step 3** `vision_agent` | `image_refs`（非空且可读） | `device_id`、`alarm_code` | ✅ 空清单 → 零成本短路（不回写任何内容，形状恒定）；单张坏图 → 该图 `basis="failed"` 占位 |

两个子图**读的都是 A 类字段**（Step 1 输入），写的都是各自那几列：

```
Step 2 写：calculated_metrics / threshold_flags / effective_alarm_codes /
          last_alarm_codes / all_alarm_codes_in_window /
          llm_description / basic_judgment / rag_search_queries
Step 3 写：visual_description / visual_findings
```

**两者写入的键完全不重叠** → 并行执行时状态合并不会冲突。

### 7.2 四种输入形态分别该跑什么（建议）

| 用户输入 | 建议编排 | 实测 |
| :--- | :--- | :--- |
| 设备 + 窗口 + 图片（完整诊断） | Step 2 **和** Step 3（串行或并行都行） | ✅ 15 个公共字段齐全 |
| **只有图片**，无窗口无设备 | **只跑 Step 3**（或照跑 Step 2，它会自动降级） | ✅ 主图跑通，Step 3 正常出结论；Step 2 三行日志说明"跳过" |
| 报警码 + 图片，无窗口 | 同上；若想用报警码补窗口，**由 Step 1 负责推窗口**（Step 2 不会自己造窗口） | ✅ 同上 |
| 简单问答（如"这铭牌流量多少"） | **只跑 Step 3**，视觉结论直接交给下游回答，不做时序分析 | ✅ 同上 |

> **Step 2 不会自己推窗口。** 如果希望"有报警码就自动查最近一次报警的时间当窗口"，
> 那是 Step 1（动态路由）或编排层的决定，做完把窗口填进 `start_time`/`end_time` 即可 ——
> Step 2 只认这两个字段。

### 7.3 编排选项与注意事项

| 选项 | 拓扑 | 收益 | 注意 |
| :--- | :--- | :--- | :--- |
| **串行（现状）** | `START → step2 → step3 → END` | 最简单，日志顺序清楚 | Step 2 缺窗口时会降级但仍占一个节点；完整诊断时多花 Step 2 那几秒 |
| **条件串行** | `START →(有窗口?)→ step2 → step3 → END`，无窗口时直接到 step3 | 缺窗口时省掉 Step 2 的节点开销 | 判据只看 `start_time`/`end_time` 是否非空（与 Step 2 内部判据一致） |
| **并行** | `START ─┬─ step2 ─┐`<br>`　　　└─ step3 ─┴─► END` | 完整诊断时省掉 Step 2 的等待时间 | ① 需要 join（两条支线都完成才继续）；② **同时打大模型 API**（Step 2 语义化 1 次 + Step 3 每图 1 次），要控制并发；③ 逐图顺序仍由 Step 3 内部保证 |

> ⚠ **不要在 Step 3 前面塞强依赖**：Step 3 不读 Step 2 的任何产出。
> 若编排上让 Step 2 失败就整图失败（例如 Step 2 抛异常），Step 3 的结论会一起丢掉 ——
> 这正是"软降级"要避免的情形。

---

## 8. 一句话总结

> **OCR 负责"看清字"，大模型负责"看懂图"，Python 负责"别做白工"。**
>
> 一条图进来：**在原图上**跑一遍 Tesseract（0.8~1.9 s，只判"噪声还是有用"）→
> 缩放到 API 限制（多数情况**原样直传**）→ **必过一次视觉大模型**（唯一的大头耗时）→
> 用 OCR 数字校正模型读数（0 ms，规则保守、改与不改都留痕）→
> 回写**两半**：一段给人读的描述（Step 4 RAG 输入）+ 一个给机器读的强类型盒子。
>
> 图片字节**从不进 State**；一张图失败**不拖垮整批**；OCR 挂掉**如实上报**而不是静默降级；
> 模型失败**重试但不退化成仅 OCR** —— 因为实测告诉我们：**省下那次调用，代价是丢掉 `FAL-104`**。
