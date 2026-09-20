# Step 3 · vision_agent 实现说明

> **这份文档回答三件事**：
> 1. 图片从上游进来、到最终出去，**中间变成了什么形状**（用真实跑批数据逐步展示）
> 2. 每个关键设计**为什么这么做**（尤其是"为什么每张图都必须过视觉大模型""为什么 OCR 只做辅助"）
> 3. **怎么提测**（含中英文 OCR 引擎的系统级安装要求）
>
> ⚠ 本模块在 2026-09-17 做过一次**精简**：原来的 2 节点子图合并成**一个节点函数**，
> 结构化输出**只保留 3 个字段**。本文档描述精简后的现状。

---

## 0. 这个模块是什么

**一句话**：把上游递来的「一批图片（路径或 URL）」，变成两份东西 —— 一份**给人读的整段描述**（给 Step 4 做 RAG 检索、给 Step 7 写报告），一份**给机器读的结构化结论**（每张图的观测 / 极性 / 未核验项）。

```
                  ┌──────────────────────────────────────────────┐
  Step 1 / 用户 ──►│  vision_agent（Step 3）                      │
  image_refs       │  vision_node(state)  ← 主图里就是一个节点     │
  (list[str])      │    逐张 analyze_one：                        │
                  │      读字节 → 判文件头 → OCR → 判噪声 →      │
                  │      缩放 → **视觉大模型** → 组装             │
                  └────────────────┬─────────────────────────────┘
                                   │ 只回写 2 个公共字段（+ 清洗后的 image_refs）
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
      visual_description                        visual_findings
   （→ Step 4 RAG 检索 / Step 7 报告）      （→ 机器读：部位/现象/极性/未核验）
```

### 文件清单

| 文件 | 行数 | 职责 | 依赖 LangGraph？ | 依赖大模型？ |
| :--- | ---: | :--- | :---: | :---: |
| `vision_nodes.py` | 81 | **编排层**：唯一节点 `vision_node`（批量 / 空图短路 / 失败隔离 / 写回契约） | ❌（普通函数） | ❌ |
| `vision_pipeline.py` | 92 | **管道层**：单图 `analyze_one`（读图→OCR→体检→缩图→调模型，不写 State） | ❌ | 间接 ✅ |
| `vision_client.py` | 155 | **模型层**：提示词 + `understand_image` + 重试（★ **唯一**碰大模型的文件） | ❌ | ✅ |
| `vision_compose.py` | 100 | **组装层**：盒子 ↔ 文本（`compose_description` / `ref_label` / `failed_image`，纯函数） | ❌ | ❌ |
| `image_io.py` | 201 | **工具层**：读图 / 判 MIME / 缩放 / 拼 data URL（`Pillow`，延迟导入） | ❌ | ❌ |
| `ocr_runner.py` | 76 | **工具层**：调 Tesseract 认字 → `OcrResult`（`pytesseract` 是**可选**依赖，没装也不抛） | ❌ | ❌ |
| `ocr_quality.py` | 136 | **工具层**：判 OCR 文本是"噪声"还是"有用"（`evaluate_ocr_quality` 是**纯函数**，零第三方依赖） | ❌ | ❌ |
| `../../schemas/vision.py` | 89 | 3 字段契约：`VisionImage` / `VisionFindings` / `Observation` | ❌ | ❌（同时是模型的输出 schema） |
| `../../../tests/test_vision_agent.py` | 493 | 27 条单测（契约 / 工具 / OCR / 编排 / 失败隔离 / 重试 / 真实图） | ❌ | 部分 ✅ |
| `../../../scripts/run_vision_samples.py` | 339 | 样例图跑批（人工验货，不进 CI） | ❌ | ✅ |

> **已分层**（2026-09-18）：原 `vision_nodes.py`（356 行）按**"职责"**拆成四层 ——
> `vision_nodes.py`（编排，只剩 `vision_node`）/ `vision_pipeline.py`（单图管道）/ `vision_client.py`（模型层）/
> `vision_compose.py`（组装，纯函数）。依赖是**单向直线**（没有回头箭头）：
> `vision_nodes → vision_pipeline → vision_client`；`vision_pipeline → vision_compose`；`vision_pipeline → 工具层`；
> 另有 `vision_client → image_io`（只为把缩放后的字节拼成 data URL）。
> 三个**叶子层**谁也不依赖别人：`image_io` / `ocr_runner` / `vision_compose`。

> **已拆分**（2026-09-18）：原 `image_tools.py`（383 行）按**"外部依赖"**这条缝拆成三个文件 ——
> `image_io.py`（要 Pillow）/ `ocr_runner.py`（要 pytesseract，且可选）/ `ocr_quality.py`（纯 Python，啥都不要）。
> 依赖方向是**单箭头**：`ocr_quality → ocr_runner`（只有 `assess_ocr(text=None)` 那一支会真去调 `run_ocr`）；
> `image_io` 谁都不依赖、也不被谁依赖。

> **已删除**（2026-09-17 精简）：`vision_graph.py`（子图定义）、`vision_state.py`（子图状态）。
> 原因：这个子图**没有私有状态**，`ingest` 只做"清洗清单 + 空图短路"（函数开头 3 行就能表达），
> 而且在链路上不需要 checkpoint、没有 interrupt —— 包成子图只是多一层跳转。

> **`image_io.py` / `ocr_runner.py` / `ocr_quality.py` / `vision_compose.py` 都不含任何 LangGraph 与大模型依赖** ——
> 它们只吃字节、只吐文本/数值/盒子，所以能脱离图和 API 单独测试。这是刻意的：
> **工具归工具，组装归组装，模型调用只收在一层（`vision_client.py`）**。

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
| 缺 `tesseract` 可执行文件 / 缺 `chi_sim` | `run_ocr()` 返回 `ok=False` + 原因 | **不崩**：如实写进 `limitations`（"OCR 未生效（…）"），图照常送模型；代价是丢了文字辅助 |
| 缺 `pytesseract` | 同上（`reason="未安装 pytesseract"`） | 同上 |
| 缺 `Pillow` | `ModuleNotFoundError` 直接抛出 | 该图变成失败占位（可溯源），**其他图不受影响** |
| 缺 `DEEPSEEK_API_KEY` | `init_chat_model` 在 import 期就报 ValidationError | 主图根本建不起来（测试会明确 skip 并说明原因） |

### 1.2 三条提测命令

```bash
# ① 契约 + 工具 + 编排 + 失败隔离（不需要网络；有 Key 时会跑 1 张真图）
$PY -m pytest tests/test_vision_agent.py -q
#    期望：27 passed
#    覆盖：3 字段契约 / MIME / data URL / 缩放快速通道 / OCR 判噪声 /
#          空图短路 / 有用文本进提示词但绝不跳过模型 / OCR 失败如实上报 /
#          失败隔离三条 / 重试两条 / 主图挂载 / 真实图 live

# ② Step2 + Step3 的契约与挂载
$PY -m pytest tests/test_nodes.py -q
#    期望：12 passed   ← 与上面合计 39 passed

# ③ 样例图跑批（**真实调 API、要花钱**，人工看识别质量）
$PY scripts/run_vision_samples.py --json /tmp/vision_report.json
#    期望：汇总：成功 9 / 失败 0
#    可用参数：--dir / --limit N / --fail-fast / --json
```

### 1.3 命令失败自查

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| 每张图都 `OCR 体检 : 噪声→丢弃 | 0 字` | Tesseract 没装好 / 缺字库 | `tesseract --list-langs`；按 §1.1 ② 重装 |
| `ModuleNotFoundError: No module named 'PIL'` | 解释器选错了（旧 venv 没装 Pillow） | 用 `/home/yali_ai/work_file/.venv/bin/python` |
| 测试全部 `SKIPPED: 需要 DEEPSEEK_API_KEY` | 当前 shell 没导出 Key | `eval "$(grep -E '^export DEEPSEEK_API_KEY=' ~/.bashrc)"` |
| **`httpx2.InvalidURL: Invalid port: ':1]'`** | 环境里的 `NO_PROXY` 含 `[::1]` 等写法，httpx2 把它当 URL 解析（**与代码无关**） | `export NO_PROXY="localhost,127.0.0.1,::1"`（或运行时 `env -u HTTP_PROXY -u HTTPS_PROXY -u NO_PROXY -u no_proxy …`） |
| `RuntimeError: 视觉结构化输出解析失败（返回 None）` | 输出被 `max_tokens` 截断（连重试 3 次都没成） | 提高 `llm_client.py` 的 `max_tokens` |
| 摘要表里某行 `状态=ERROR` | 该图处理失败（路径写错/格式不支持） | 看该行打印的失败原因 |
| 跑批很慢（每图 5~10 s） | 正常：一次 OCR + 一次视觉 API | 见 §2 耗时分解；批内并行尚未实现（§6-1） |

> `scripts/run_vision_samples.py` 是**人工验货工具**，不是 pytest 用例：
> 它对每张图单独 try/except（生产代码里这层兜底在 `vision_node` 内），失败只打印不断言，退出码 1 表示有图失败。

---

## 2. 完整数据流

### 📊 全局一览（真实批量跑批，9 张样例图）

| 阶段 | 谁在算 | 数据形态 | 单图耗时 |
| :--- | :--- | :--- | :--- |
| 0 输入 | Step 1 / 用户 | `image_refs`（list[str]） | — |
| 1 读字节 + 判格式 | Python | `bytes` → MIME 字符串 | 1 ms |
| 2 OCR（1 遍） | Tesseract | 一段文本（可能为空） | **0.8~1.9 s** |
| 3 判噪声/有用 | Python 正则 | 一个布尔值 | 0 ms |
| 4 缩放 | Pillow | 多数情况**原字节直传** | **0 ms** |
| 5 **视觉大模型** | DeepSeek | 3 字段结论 | **4.3~6.1 s（74~84%）** |
| 6 拼文本 | Python | `[图N] …` 段落 | 0 ms |
| 7 输出 | 图回流 | 2 个公共字段 | — |

### 阶段 0 ｜输入：Step 1 / 用户写进父图 `DiagnosisState`

```json
{
  "device_id": "PUMP-IS100-80-160-01",
  "alarm_code": "FAL-104",
  "image_refs": ["data/sample_inputs/vision/leak_close.png",
                 "data/sample_inputs/vision/hmi_alarm.png"]
}
```

Step 3 只需要其中 **3 个**：`image_refs`（要处理什么）与 `device_id` / `alarm_code`（填提示词）。
**窗口、阈值、报警码的判定全都不归它管。**

### 阶段 1 ｜`vision_node` 开头：清洗清单 + 空图短路

```python
refs = [str(r).strip() for r in state.get("image_refs") or [] if str(r).strip()]
if not refs:
    return {"image_refs": [], "visual_description": "", "visual_findings": VisionFindings()}
```

**没图 → 不读图、不 OCR、不调模型，成本为零**（这也替代了原来单独一个 `ingest` 节点）。

### 阶段 2 ｜逐张 `analyze_one`（★ 顺序本身就是设计）

| # | 调用 | 真实输入 → 输出 | 耗时 |
| :---: | :--- | :--- | ---: |
| 1 | `read_image_bytes` | 路径/URL → 原始字节（本地 / http(s) 都在这里判） | 1 ms |
| 2 | `detect_content_type` | **文件头魔数** → `"image/png"`；不是 JPG/PNG/GIF/WebP 就**在此抛出** | 0 ms |
| 3 | `run_ocr`（**在原图上**） | 图 → `OcrResult(text, ok, reason)` | 0.8~1.9 s |
| 4 | `assess_ocr` | 文本 → `credible`（噪声 or 有用） | 0 ms |
| 5 | `resize_for_api` | 1.27MB / 1672×941 **已在限制内 → 原样直传** | **0 ms** |
| 6 | **`understand_image`**（`vision_client.py`） | 图 + OCR 文本 → 3 字段结论（带重试） | 4.3~6.1 s |
| 7 | 组装 `VisionImage`（`vision_compose.py`） | + OCR 没生效时追加 limitations | 0 ms |

> 上表 1~5 步全在**工具层**（`image_io.py` / `ocr_runner.py` / `ocr_quality.py`），第 6 步在**模型层**（`vision_client.py`）；
> 整条管道由**管道层** `vision_pipeline.py::analyze_one` 串起来，第 7 步的拼装归**组装层** `vision_compose.py`。

**第 2 步的位置很关键**：格式校验在 **OCR 之前**。实测喂一个假 `.png`（其实是文本）：

```
失败于: ValueError: 不支持的图片格式（需要 JPEG / PNG / GIF / WebP）
耗时: 0.1 ms        ← 若真跑了 OCR 会是 800+ ms，说明 OCR 没有被白跑
```

**第 3 步的真实输出**（Tesseract 原文，节选，HMI 截图）：

```
| |  Pump Monitoring System
Centrifugal Pump Group
四           Trend          |          Alarms        Reports      Settings
12.5 WL    8.9 m?/h
0.25  MPa
Ga Pump P-101  | ”Model 1IS100-80-160  Location Pump Room 1   Status RUN
```

> **看出来了吗**：`IS100-80-160` 被认成 `1IS100-80-160`、`m³/h` 变成 `WL`/`m?/h`，
> 而且**整条 `ALARM FAL-104 LOW FLOW` 告警横幅根本没被认出来**。
> 这就是"OCR 只做辅助、每张图必须过模型"的实测依据（见 §3.4）。

**第 6 步的真实输出**（模型给的 3 字段结论，节选）：

```jsonc
{ "natural_description": "这是一张离心泵监控系统（Pump Monitoring System）的 HMI 截图，
    顶部为红色告警条，显示 ALARM FAL-104 LOW FLOW，时间戳 2026-09-13 00:41:07…",
  "observations": [
    {"target": "顶部告警条", "finding": "红色告警条显示 ALARM FAL-104 LOW FLOW…", "polarity": "abnormal"},
    {"target": "Flow 卡片 PV 过程值", "finding": "显示 PV 8.9 m³/h，低于同卡片 SP 12.5 m³/h", "polarity": "abnormal"},
    {"target": "右上角运行状态指示", "finding": "绿色圆点与 RUN 字样", "polarity": "normal"}],
  "limitations": ["截图为 HMI 界面，非现场实拍，无法看到泵体、管路、密封等物理外观",
                  "趋势图曲线为像素级目测读数，未标注数据点",
                  "OCR 摘录中部分字符存在识别错误，已按画面实际显示校正"] }
```

### 阶段 3 ｜输出：回写 3 个键

```
节点写回的键（其余是上游已有的）：
  image_refs          清洗后的清单（★ 为了让 images[i] ↔ image_refs[i] 等长同序）
  visual_description  人读：每图一段 + [图N] 图名 + "本图未核验：…"
  visual_findings     VisionFindings(images=[…])  ← 每图 3 个字段
```

**`visual_description` 的真实内容**（由 `visual_findings` 派生，唯一写入者）：

```
[图1] leak_close.png
画面为卧式离心泵组（灰黑色泵体 + 蓝色电机与轴承箱）的近景。泵体与轴承箱之间的填料压盖 /
密封压盖区域可见明显的深色油性液体，沿压盖下缘连续向下流淌，在泵体下方的底座与基础面上
形成一滩积液，液面反光。
本图未核验：无 OCR 结果，画面中未见铭牌、表计读数等文字信息，无法提供任何数值读数；
泄漏液体的具体性质（油/水/介质）无法从画面判断；泄漏起始时间与泄漏速率无法从单张静态图像判断
```

### 📊 9 张样例图跑批（真实结果，一次代表性运行）

| 图片 | 状态 | 观测条数 | abnormal | 说明 |
| :--- | :--- | ---: | ---: | :--- |
| `leak_close.png` | OK | 6~7 | 2~3 | 密封压盖渗漏（核心用例） |
| `leak_wide.png` | OK | 8 | 2 | 泵组全景 + 地面湿渍 |
| `gauge_low.png` | OK | 5~7 | 0~3 | 压力表读数 |
| `hmi_alarm.png` | OK | 11 | 3 | 唯一 OCR 有用的一张（文本进了提示词） |
| `nameplate.png` | OK | 10 | 0 | 铭牌参数全对（OCR 只抓到部分） |
| `thermal_bearing.png` | OK | 6 | 0 | 热成像 |
| `bearing_rust.png` | OK | 8 | 4~6 | 锈蚀/渗油 |
| `bad_quality.png` | OK | 7~8 | 0 | 逆光+模糊：未核验项最多（6 条） |
| `normal_control.png` | OK | 9 | **0** | 正常对照组：**没有误报异常** |

> 观测条数 / abnormal 条数在多次运行间有 ±1~2 条波动（模型侧固有波动，`temperature=0` 也不能完全消除）。

---

## 3. 关键设计决策与理由

### 3.1 为什么是**一个节点函数**，不是子图？

原来的 2 节点子图（`ingest → extract`）满足不了"子图"的三个存在理由：

| 子图的常见理由 | 本模块的实际情况 |
| :--- | :--- |
| 有私有状态要跨节点传 | ❌ `VisionAgentState` 与公共契约**完全等价**（零私有字段） |
| 需要自己的 checkpointer / interrupt | ❌ 线性流程，不需要暂停恢复 |
| 节点多、图能表达复杂编排 | ❌ 只有一条直线，`ingest` 就是"清 3 行 + 判空" |

所以合并成 `vision_node(state)`：**少一层跳转、少两个文件，能力不减**。
主图里就是一行 `main.add_node("step3", vision_node)`。

> 实测还验证过：给这个子图单独加 checkpointer **对主图的 checkpoint 流毫无影响**
> （条数与命名空间完全一致）—— 即"给子图加记忆"是个无效动作。

### 3.2 为什么图片字节**不进 State**？

节点之间/父子图之间只传 `image_refs`（字符串清单），字节是 `analyze_one` 内部的局部变量。

**理由**：LangGraph 的 State 每经过一个节点都要**整份合并一次**。几百 KB ~ 几 MB 的字节进去，
每次状态合并都在拷贝它。而它唯一的用途就是"喂给 API 一次"，用完即弃。
**副产品**：State 里没有任何 bytes 字段 → 契约干净、序列化/SQLite checkpointer 都不会爆。

### 3.3 为什么格式校验要放在 **OCR 之前**？

`analyze_one` 的第 1、2 步就是 `read_image_bytes` + `detect_content_type`（按魔数），
第 3 步才 OCR。所以**坏格式在花钱、花时间之前就被拦下**（实测 0.1 ms）。

- 报错文案是**可执行**的：`不支持的图片格式（需要 JPEG / PNG / GIF / WebP）`
- 该图随后变成**失败占位**（`natural_description` 以 `（本图识别失败` 开头 + 原因写进 `limitations`）
- ⚠ "对用户回话"（"请上传 JPG/PNG"）**不在这里做** —— 那是最后一站（Step 7 / 接口层）的事；
  这里只负责"如实记录哪张没成、为什么"，**并且不影响其他图**。

### 3.4 ★ 为什么**每张图都必须过视觉大模型**（不省钱）？

| 信息 | Tesseract（OCR） | 视觉大模型 |
| :--- | :--- | :--- |
| **关键告警码 `FAL-104`** | ❌ **完全没认出来** | ✅ 认出来了 |
| 型号 `IS100-80-160` | ❌ 认成 `1S100-80-160` | ✅ 正确 |
| 单位 `m³/h` | ❌ 认成 `m?/h`、`WL` | ✅ 正确 |
| 铭牌 `100 m³/h` / `32 m` | ❌ **漏掉**（只抓到 `2900 r/min`、`15 kW`） | ✅ 全对 |
| 速度 | **快**（1~2 s） | 慢（5~6 s + 花钱） |

**结论**：OCR 的价值在"快"和"当交叉核对"，**不在"准"**。省下那一次模型调用，
代价是把 `FAL-104` 这种诊断起点弄丢 —— 不值。
所以"**OCR 独证**（文字够完整就不调模型）"这条路已被整体删除。

### 3.5 为什么 OCR 的判定只有"**噪声 / 有用信息**"两个结论？

```
credible = 字符数 ≥ 24  且  数字个数 ≥ 4  且  (命中单位 / 位号 / 字段名 之一)
```

| 判据 | 挡掉什么 |
| :--- | :--- |
| 字符数 ≥ 24 | 零散噪声（真实照片常吐 `"ww Wink"` 这种 7 个字符的东西） |
| 数字 ≥ 4 | 纯字母/纯符号噪声（我们要的是读数与位号） |
| 单位/位号/字段名 | 证明这段文字**属于工业语境**，不是随机纹理 |

> 行数、Tesseract 置信度**不参与判断**（已从代码里删除）：实测热成像刻度数字的
> 平均置信度高达 94，却只有 22 个字 —— 只看置信度会被骗。

### 3.6 为什么 `resize_for_api` 要有一条"**已在限制内就原样返回**"的快速通道？

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

### 3.7 为什么删掉"第二遍 Tesseract"？

`assess_ocr` 曾经为了拿"行数 + 平均置信度"再调一次 `pytesseract.image_to_data` ——
**同一个引擎、同一张图、跑第二遍**，实测 **0~1.7 秒/图（占 12%）**，而那两个数**不参与任何判断**。

现在：`_text_layout()` 整个函数删除，`OcrAssessment` 去掉 `lines` / `confidence` 两个字段。

### 3.8 为什么**禁止清洗 OCR 文本**？

1. **收益只有 token**：实测清洗省 24% 字符，但**对"噪声/有用"判断零影响**（判据本来就把空白剥掉了）。
2. **字符串里的"排版"多半是误识别**：HMI 那张的 `| |  Pump`、`四` 是 Tesseract 把界面边框/图标认成了字符
   （逐词置信度 24/44/12，而真文字都是 90+）。
3. **风险不对称**：清洗规则一旦存在很容易被"顺手加强"，而某张图上那恰好是内容。

> 真正的排版信息在 `image_to_data` 的 x/y 坐标里，不在字符串里。

### 3.9 为什么一张图失败不能拖垮整批？

`vision_node` 里每张图**独立** `try/except`，失败产出一条**占位结论**：

```jsonc
{ "natural_description": "（本图识别失败，未产生视觉结论：FileNotFoundError: …）",
  "observations": [],
  "limitations": ["处理失败：FileNotFoundError: …"] }
```

**为什么是占位而不是"跳过这张"**：批量识别时其他图的结论是真金白银换来的；
而且失败必须**可溯源** —— 靠 `[图N] 图名` 锚点 + 这段文字，读者能看出"第 N 张没识别成功、原因是什么"。
下游判定失败的方式：`image.natural_description.startswith(FAILED_PREFIX)`（模块常量）。

### 3.10 为什么重试只有**一层**？

原来实际是**双层**：底层 OpenAI SDK 默认 `max_retries=2`，外面再 3 轮 → 最坏 **9 次真实 HTTP 请求**
（每次都是完整图文请求、重复计费），日志里却只看到"第 3 次失败"。

**现在**：`vision_model` 显式 `max_retries=0`，重试权收归 `understand_image`（`vision_client.py`）一层：

| 情况 | 行为 |
| :--- | :--- |
| 网络波动 / 连接重置 / 超时 / 限流 / 5xx | **重试**，最多 3 次（含首次），退避 **1s → 2s** |
| 结构化输出返回 `None`（多为截断） | 也重试；用尽才报错 |
| **余额不足**（`Insufficient Balance` / `402`） | **立刻抛**，不重试、不等待 |
| 鉴权失败 / 模型名写错 / 请求不合法 | 立刻抛 |

判定方式用"黑名单"：只列"确定重试无用"的几类，宁可多等几秒，也别把一次本来能成功的识别判死。
同文件里还有 `timeout=60`（SDK 默认读超时是 600 秒，一次"连上但不回数据"能卡 10 分钟）。

> ⚠ 文本模型 `model`（Step 2 用）**保持不动**：Step 2 没有自己的重试层。

### 3.11 为什么契约只有 **2 个公共字段 + 每图 3 个字段**？

| 公共字段 | 消费者 | 形态 |
| :--- | :--- | :--- |
| `visual_description` | **人**：Step 4 生成 RAG 检索词、Step 7 写报告 | 一段文本（`[图N]` 锚点 + `本图未核验`） |
| `visual_findings` | **代码**：遍历/计数/过滤 | `VisionFindings(images=[VisionImage])` |

**每图只有 3 个字段**（2026-09-17 精简，除这三个之外全删）：

| 字段 | 含义 |
| :--- | :--- |
| `natural_description` | 这张图的人话描述（也是文本派生的源头） |
| `observations[]` | `target`（部位）/ `finding`（看见了什么）/ `polarity`（abnormal / normal / unknown） |
| `limitations` | 没看清什么（"未核验"清单）；OCR 未生效、处理失败的原因也写在这 |

**曾经有、现在删掉的字段**：`source_ref`（改用**位置对齐**：`images[i] ↔ image_refs[i]`，
所以 `vision_node` 会回写清洗后的 `image_refs`）、`image_kind`、`basis`、`confidence`、
`quantity`（连带删除了 OCR 数字校正）、`evidence`。

**关键规则：文本只能由盒子派生，且只有 `compose_description`（`vision_compose.py`）一个写入者。**
只要写入者唯一，两份表示就不是"两处真相"，而是"同一处真相的两种视图"，永不漂移（有测试守护）。

### 3.12 为什么提示词里**不写 JSON 格式**，且"只描述不判断"？

- **不写格式**：用了 `with_structured_output(VisionImage)` —— **schema 由 LangChain 自动注入**
  （实测模型看到的字段就是那 3 个；枚举 `polarity` 也是注入的，模型填不出别的值）。
- **只描述不判断**：system prompt 明确"**禁止输出「疑似 XXX 故障」「原因是……」等归因**"。
  归因是 Step 5 的事；Step 3 只交"看得见的现象"。实测模型会自我纠正 OCR 错字并写进 `limitations`。

### 3.13 为什么 `limitations` 保持自由文本？

三个来源：① **主来源**是模型自述（提示词要求"看不清、反光、被遮挡的内容写入 limitations，不要编造"）；
② 代码在 `run_ocr` 失败时追加"OCR 未生效（原因）"；③ 处理失败时写"处理失败：<异常类型>: <消息>"。

已知边界：无校验、无条数上限；**条数不是质量信号**（它是按提示词"写齐"的）；
自由文本、无分类，代码无法区分"遮挡/分辨率/角度/参数缺失"。**改成机器信号（枚举分类）等有消费者时再做。**

---

## 4. 测试用例清单（27 条）

跑法：`$PY -m pytest tests/test_vision_agent.py -q` → 期望 `27 passed`
（连同 `tests/test_nodes.py` 的 12 条，合计 **39 passed**）

| 组 | 条数 | 覆盖意图 |
| :--- | ---: | :--- |
| **契约** | 2 | 公共字段形状；**每图恰好 3 个字段**、观测恰好 3 个字段（防回退回潮） |
| **工具（无 API）** | 5 | MIME 判型（PNG/JPEG/垃圾）；data URL 前缀；缩放快速通道；超限仍缩 |
| **OCR 判噪声/有用** | 5 | 真实 HMI 有用；理想文字屏有用；`"ww Wink"` 是噪声；无单位/位号 → 噪声；无数字 → 噪声 |
| **OCR 结果语义** | 2 | 空文本不跑 Tesseract；`OcrResult` 区分"图上没字"与"OCR 没跑成" |
| **节点编排（替身）** | 3 | 空图全短路；有用文本进提示词但**绝不跳过模型**；OCR 挂掉仍走模型且如实上报 |
| **文本派生** | 1 | `compose_description`：锚点 / limitations 折叠 / 无 limitations 不出那行 |
| **失败隔离** | 3 | 一张坏图不影响好图；失败原因在盒子与文本里；全坏时形状恒定 |
| **重试** | 3 | 单层（两处 `max_retries==0`）；网络波动重试成功；余额不足不重试不等待 |
| **主图 + 真实图** | 3 | 挂在主图；父图只见契约字段；真实样例图跑通（1 次 API） |

### 三个值得单独说的用例

**① `test_batch_keeps_good_images_when_one_fails` —— 失败隔离回归**
```
输入 3 张：["good.png", "坏图.png", "good.png"]
期望：结论条数 == 3；坏的以 FAILED_PREFIX 开头；两张好图的结论都还在文本里
```

**② `test_box_has_exactly_three_fields_per_image` —— 契约防回潮**
```
期望：VisionImage 字段集合 == {natural_description, observations, limitations}
      Observation 字段集合 == {target, finding, polarity}
```
> 这条守护"精简后的 3 字段不要再涨回去"。

**③ `test_useful_ocr_text_is_passed_to_model_but_never_skips_it` —— "辅助而非替代"**
```
monkeypatch 记录模型收到的 ocr_text
期望：提示词里含 "ALARM FAL-104"；描述来自模型；polarity 正常透传
```

---

## 5. 怎么新增/修改测试用例

| 类型 | 需要什么 | 写法 |
| :--- | :--- | :--- |
| **纯函数/工具测试** | 不需要 Key、不需要网络 | 直接调 `evaluate_ocr_quality(...)` / `resize_for_api(...)` / `detect_content_type(...)` |
| **节点编排测试** | 需要 Key（`vision_nodes` 在 import 期就构造模型） | 加 `@requires_key`；用 `monkeypatch` 换掉 `run_ocr` / `vision_model` / `analyze_one`，**不打真 API** |
| **真实图 live** | 需要 Key + `data/sample_inputs/` 下有图 | 只挑 1 张（批量请用跑批脚本，别塞进 pytest —— 那是真花钱的） |

**约定（重要）**：

1. **测试替身不写进生产代码** —— 假模型用 `monkeypatch` 注入（文件里的 `_fake_model()` 是个现成小工具）。
2. **不要再用"吞掉异常 → skip"**：只有两个前置条件缺失才允许 skip（没 Key / 没样例图），
   **节点建不起来、调用失败都必须红**。
3. **断言口径：绑语义，不绑数据指纹。**
   ✅ `"ALARM FAL-104" in seen["ocr_text"]`、`natural_description.startswith(FAILED_PREFIX)`
   ❌ 绑死某次运行恰好有多少条观测（模型侧有固有波动）

---

## 6. 已知边界

| # | 边界 | 影响 | 现状 |
| :---: | :--- | :--- | :--- |
| 1 | **多图串行** | 9 张图 ≈ 9 × 7 s ≈ 60 s（每次调用彼此独立、且是 I/O 等待） | 未做并行。`configs/config.yaml` 已有 `concurrency.max_workers: 4`，用线程池预计可降到 **~15 s** |
| 2 | **OCR 对真实截图不够准** | 漏认 `FAL-104`、`IS100`→`1S100`、`m³/h`→`m?/h`；铭牌漏掉 `100 m³/h` 与 `32 m` | 这是"每图必过模型"的直接原因（§3.4）。模型会自行按画面校正并写进 `limitations` |
| 3 | **失败状态没有专用字段** | `basis` 已删，判定"这张失败"只能靠 `natural_description.startswith(FAILED_PREFIX)` | 可工作但不优雅；若下游需要，最小修法是加回一个 `status` 字段 |
| 4 | **OCR 是否参与、图片种类不可见** | `basis` / `image_kind` / `confidence` 已删，下游无法区分"OCR 帮了忙"与"纯看图" | 目前只体现在节点日志里；有消费者时再加字段 |
| 5 | **`limitations` 是自由文本、无校验、无上限** | 无法用代码区分"遮挡/分辨率/角度/参数缺失"；条数与"问题多少"无关 | 保持现状，等有消费者再上枚举 |
| 6 | **演示图的时间戳是编的** | `hmi_alarm.png` 画面写 `00:41:07`，而库里 FAL-104 真实发生在 `10:30` —— 跨模态时间对不上 | 生成图片时请用真实窗口；跨模态时间对齐是 Step 5 的事 |
| 7 | **`configs/config.yaml` 的 `vision:` 段是死配置** | 没人读它（模型参数都在 `src/utils/llm_client.py`） | 待收敛 |
| 8 | **环境里的 `NO_PROXY` 可能让模型客户端建不起来** | `httpx2.InvalidURL: Invalid port: ':1]'`（本地代理客户端写入的 `[::1]` 等写法） | 与代码无关；修法：`export NO_PROXY="localhost,127.0.0.1,::1"`，或运行时 unset 代理变量 |
| 9 | **文本模型（Step 2）没有自己的重试层** | 它依赖 SDK 默认 `max_retries=2` | 刻意不动（属 Step 2 改动范围） |

---

## 7. 给编排方（Step 1 / 组长）的对接说明

> **本节是写给"决定主图怎么排"的人看的**：不需要读代码，看下面三张表就能拼。
> Step 2 与 Step 3 **互相独立**（谁都不读对方的产出），所以它们可以**串行、并行、或各自跳过**。

### 7.1 两个节点需要什么、给出什么

| 节点 | 必须有 | 可选 | 缺"必须有"时的行为 |
| :--- | :--- | :--- | :--- |
| **Step 2** `data_agent`（子图） | `device_id` + `start_time` + `end_time` | 无（★ 2026-09-17 起 Step 2 **不再读** `alarm_code`，报警码一律取自窗口数据） | ✅ **软降级**：不查库、不计算、不调大模型，返回空结果 + 告警 `未提供时间窗口：本次未做时序分析` |
| **Step 3** `vision_node`（节点函数） | `image_refs`（非空且可读） | `device_id`、`alarm_code` | ✅ 空清单 → 零成本短路（形状恒定）；单张坏图 → 该图失败占位 |

两者读的都是 A 类字段，写的键**完全不重叠**：

```
Step 2 写：calculated_metrics / threshold_flags / llm_description
          （★ 2026-09-17 精简：effective_alarm_codes / last_alarm_codes /
            all_alarm_codes_in_window / basic_judgment / rag_search_queries
            五个字段已从公共契约删除；窗口内出现过的报警码改为
            calculated_metrics.overall.effective_alarm_codes）
Step 3 写：image_refs（清洗后）/ visual_description / visual_findings
```

### 7.2 四种输入形态分别该跑什么（建议）

| 用户输入 | 建议编排 | 实测 |
| :--- | :--- | :--- |
| 设备 + 窗口 + 图片（完整诊断） | Step 2 **和** Step 3（串行或并行都行） | ✅ 公共字段齐全 |
| **只有图片**，无窗口无设备 | **只跑 Step 3**（或照跑 Step 2，它会自动降级） | ✅ 跑通，Step 3 正常出结论 |
| 报警码 + 图片，无窗口 | 同上；若想用报警码补窗口，**由 Step 1 负责推窗口** | ✅ 同上 |
| 简单问答（如"这铭牌流量多少"） | **只跑 Step 3**，视觉结论直接交给下游回答 | ✅ 同上 |

> **Step 2 不会自己推窗口。** 想让"有报警码就自动查最近一次报警时间当窗口"，那是 Step 1（动态路由）
> 或编排层的决定，做完把窗口填进 `start_time`/`end_time` 即可。

### 7.3 编排选项与注意事项

| 选项 | 拓扑 | 收益 | 注意 |
| :--- | :--- | :--- | :--- |
| **串行（现状）** | `START → step2 → step3 → END` | 最简单，日志顺序清楚 | 完整诊断时多花 Step 2 那几秒 |
| **并行** | `START ─┬─ step2 ─┐`<br>`　　　└─ step3 ─┴─► END` | 总耗时 ≈ `max(step2, step3)` | ① 需要 join（都完成才继续）；② **同时打大模型 API**，要控并发；③ 日志会交错（`[Node*]` vs `[Vision]` 前缀可区分） |

**不需要写条件边**：两个节点在缺输入时都会**自己安全退出**（Step 3 空清单 0 ms、Step 2 缺窗口三行日志），
所以主图只要决定"挂不挂它们"即可。

> ⚠ **不要在 Step 3 前面塞强依赖**：Step 3 不读 Step 2 的任何产出；
> 若编排上让 Step 2 抛异常导致整图失败，Step 3 的结论会一起丢掉 —— 这正是"软降级"要避免的情形。

---

## 8. 一句话总结

> **OCR 负责"看清字"，大模型负责"看懂图"，Python 负责"别做白工"。**
>
> 一条图进来：**判文件头**（坏格式 0.1 ms 就拦下）→ **在原图上**跑一遍 Tesseract（只判"噪声还是有用"）→
> 缩放到 API 限制（多数情况**原样直传**）→ **必过一次视觉大模型**（唯一的大头耗时）→
> 回写**两半**：一段给人读的描述（Step 4 RAG 输入）+ 一个给机器读的 3 字段盒子。
>
> 图片字节**从不进 State**；一张图失败**不拖垮整批**；OCR 挂掉**如实上报**而不是静默降级；
> 模型失败**重试但不退化成仅 OCR** —— 因为实测告诉我们：**省下那次调用，代价是丢掉 `FAL-104`**。
