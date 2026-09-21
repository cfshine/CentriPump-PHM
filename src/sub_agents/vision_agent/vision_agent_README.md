# Step 3 · vision_agent 实现说明

> **三句话导览**：
> 1. 输入是 `state.context.image_refs`（`list[ImageRef]`），输出**只回写 `state.vision`**；
> 2. Step 3 是**一个普通节点函数** `vision_node`（不是子图），逐张图调用视觉大模型；
> 3. 每张图**必须**过模型；OCR 只做辅助（只判"噪声还是有用"），且从不代替模型下结论。
>
> 契约版本：对齐组长 9 盒子契约（`origin/main`）；`.py` 共 8 个文件、约 870 行。

---

## 0. 这个模块是什么

**一句话**：把「一批现场图片」，变成 `state.vision` 盒子里的三样东西 —— 状态、缺陷台账、人读摘要。

```
                        ┌──────────────────────────────────────────────┐
   state.context    ──► │  vision_agent（Step 3）                       │
   └ image_refs         │  vision_node(state)  ← 主图里就是一个节点      │
     (ImageRef[])       │    逐张 analyze_one：                         │
                        │      读字节 → 判文件头 → OCR → 判噪声 →       │
                        │      缩放 → **视觉大模型** → 组装              │
                        └────────────────┬─────────────────────────────┘
                                         │ 只回写 vision 一个盒子
                                         ▼
                              state.vision（VisionState）
              ┌──────────────────────┬──────────────────────┐
              ▼                      ▼                      ▼
           status                findings                summary
     NO_IMAGE / FAILED /   每个缺陷一条 VisionFinding   人读：每图描述
     NO_DEFECT /           (image_id / defect_type /    + 末尾"未核验汇总"
     DEFECT_FOUND           severity / location /       （→ Step 4 RAG / Step 7 报告）
                            confidence / evidence)
```

### 文件清单

| 文件 | 行数 | 职责 |
| :--- | ---: | :--- |
| `vision_nodes.py` | 134 | **编排层**：唯一节点 `vision_node`（批量 / 空图短路 / 失败隔离 / 缺陷映射 / 写回） |
| `vision_pipeline.py` | 92 | **管道层**：单图 `analyze_one`（读图 → OCR → 体检 → 缩图 → 调模型） |
| `vision_client.py` | 138 | **模型层**：提示词 + `understand_image` + 重试（★ **唯一**碰大模型的文件） |
| `vision_compose.py` | 115 | **组装层**：`compose_summary` / `ref_label` / `failed_image`（纯函数） |
| `vision_models.py` | 80 | **数据模型**：`VisionImage` / `Observation`（含缺陷四要素）/ `Severity` |
| `image_io.py` | 201 | **工具层**：读图 / 判 MIME / 缩放 / 拼 data URL（Pillow，延迟导入） |
| `ocr_runner.py` | 76 | **工具层**：调 Tesseract 认字 → `OcrResult`（可选依赖，没装也不抛） |
| `ocr_quality.py` | 131 | **工具层**：判 OCR 文本是"噪声"还是"有用"（纯函数） |

> ★ `vision_models.py` 是 **Step 3 内部**的数据模型，**不属于主图公共契约**
> （公共契约在 `src/schemas/state.py`）。它同时充当视觉大模型的输出 schema。

---

## 1. 怎么提测

```bash
cd /home/yali_ai/work_file/LLM_Project/CentriPump-PHM
PY=/home/yali_ai/work_file/.venv/bin/python
eval "$(grep -E '^export DEEPSEEK_API_KEY=' ~/.bashrc)"

$PY -m pytest tests/test_vision_agent.py -q     # 本模块 → 期望 27 passed
$PY -m pytest tests/ -q                         # 全量   → 期望 57 passed
$PY -m scripts.run_vision_samples               # 人工验货：9 张样例图跑批（真花钱，不进 CI）
```

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| OCR 相关用例 skip / 限制项里出现"OCR 未生效" | 没装 Tesseract 或没装中文语言包 | `tesseract --list-langs` 看有没有 `chi_sim`；没装也能跑，只是少了文字辅助 |
| `ModuleNotFoundError: PIL` | 解释器用错（旧 venv 没有 Pillow） | 用 `/home/yali_ai/work_file/.venv/bin/python` |
| 用例报 `'DiagnosisState' object has no attribute 'get'` | 传了旧的扁平字段/dict | 用 `create_initial_state(...)` 造真实状态，读 `state.context.*` |

---

## 2. 单图管道 `analyze_one`：处理顺序本身就是设计

| 步 | 做什么 | 为什么放这个位置 |
| :---: | :--- | :--- |
| ① | 读字节（本地路径或 http(s) URL） | — |
| ② | **判文件头（魔数）** 确认 JPG/PNG/GIF/WebP | **在 OCR 之前**：坏格式在这一步就被拦下（实测 0.1 ms），不白跑一遍 Tesseract |
| ③ | 在**原图**上跑 OCR | 字越小越怕缩放，先从最高清像素里认字 |
| ④ | OCR 体检：**噪声还是有用的信息** | `ocr_quality` 是纯函数，只回答这一个问题 |
| ⑤ | 有用 → 文本进提示词；噪声/没有 → 不喂垃圾给模型 | OCR 只做辅助，模型始终要看图 |
| ⑥ | 缩放到 API 体积限制（已在限内则原样直传） | 不重新编码，省一次质量损失 |
| ⑦ | **调视觉大模型**（必过，不可跳过） | 这是唯一的下结论环节；重试也只在 `vision_client` 这一层 |
| ⑧ | OCR 没跑成 → 如实写进 `limitations` | 不静默降级：结论全部来自视觉模型这件事要留痕 |

---

## 3. 从结论到 state：缺陷映射 + 状态派生

### 3.1 缺陷映射（`abnormal` 观测 → `VisionFinding`）

```
模型返回的 Observation（Step 3 内部）             组长的 VisionFinding（进 state）
─────────────────────────────────────────       ──────────────────────────────────
target   = "泵体机械密封处"                   →  location   = "泵体机械密封处"
finding  = "压盖下缘有深色液体连续流淌"        →  （留在 summary 的叙述里）
polarity = "abnormal"                        →  （只有 abnormal 才升级为 finding）
defect_type = "LEAK"                         →  defect_type = "LEAK"
severity    = "MODERATE"                     →  severity    = "MODERATE"
confidence  = 0.85                           →  confidence  = 0.85
evidence    = "压盖下方可见连续液流痕迹…"      →  evidence    = "压盖下方可见连续液流痕迹…"
（来自哪张图）                                →  image_id    = ImageRef.image_id
```

**只有 `polarity == "abnormal"` 的观测升级为 finding**；`normal`（"外观正常"）与
`unknown`（"看不清"）不是缺陷，只留在 `summary` 里。

### 3.2 状态四态（`vision.status`）

| 情况 | status |
| :--- | :--- |
| `context.image_refs` 为空 | `NO_IMAGE` |
| 有图，但**每一张**都处理失败 | `FAILED` |
| 至少一张成功，且发现了缺陷 | `DEFECT_FOUND` |
| 至少一张成功，且没发现缺陷 | `NO_DEFECT` |

> `NO_DEFECT` 与 `FAILED` 必须分开：**"看过了没毛病"和"压根没看成"** 对下游推理的含义完全不同。

### 3.3 `summary` 长什么样

```
[图1] leak_close.png
画面为卧式离心泵机组近景……压盖下缘及支座表面覆盖大片深褐色油污，
并有液体沿支座表面连续向下流淌至底座……

未核验汇总：
- [图1] leak_close.png：泵体与底座表面反光较强，油污覆盖区域细节被遮挡；
  无法从本图判断泄漏液体的具体介质与来源部位
- [图2] hmi_alarm.png：本图为 HMI 截图，不含实体设备影像；趋势图纵轴刻度较密，只能估读
```

每图的描述由 `compose_description` 派生，末尾那节"未核验汇总"由 `compose_summary` 追加。
**`compose_summary` 是 `vision.summary` 的唯一写入者** —— 文本永远由结构化结论派生，
两者不会各说各话。

---

## 4. 关键设计决策

### 4.1 为什么 OCR 只做辅助、每张图都必须过模型？

表计读数、铭牌文字能靠 OCR 拿到，但**"压盖下缘有液体流淌"这类现象 OCR 给不了**。
反过来，光看图也可能漏掉小字。所以分工是：**OCR 提供文字线索，模型负责下结论**。
本项目**没有**"只用 OCR 就出结论"的路径（拿 OCR 冒充视觉结论比明确失败更危险）。

### 4.2 为什么只把 `abnormal` 升级为 `VisionFinding`？（用户决策 D1）

组长的 `VisionFinding` 定义是"**单条视觉缺陷发现**"。`normal` / `unknown` 不是缺陷 ——
把它们也塞进 findings 会让缺陷台账里混进"我看不清"这种条目，下游按条数统计时就失真了。

### 4.3 为什么 `limitations` 要并进 `summary`？（D2）

组长的 `VisionState` 只有 `summary` 一个文本位，而"没看清什么"必须留下 ——
否则下游会把"**没看见缺陷**"和"**没看清**"混为一谈。所以 `compose_summary` 在每图描述
之后追加一节"未核验汇总"。

### 4.4 为什么缺字段要"跳过并留痕"？（D3）

`abnormal` 但模型漏给了 `defect_type` / `severity` / `evidence` 时，三种选择：
硬报错（整张图作废）、补默认值（凭空造一个严重程度）、跳过并记进"未核验"。
**选第三种**：宁可少一条，不可编一条；而且缺的那条会在 `summary` 里明确写出来，人能看到。

### 4.5 为什么 `severity` 由模型判定、`confidence` 是"模型的把握"？（D4/D5）

- `severity` 三档口径写在提示词里（**MINOR** 轻微局部 / **MODERATE** 明显但不危及运行 /
  **SEVERE** 严重或危及运行，各附例子）；三档由**模型**按画面判断，不由代码硬编码阈值。
- `confidence` 是**模型对自己这个判断的把握**（0~1）。提示词里明确写了
  "**不是故障概率，也不是最终诊断置信度**" —— 免得它被当成故障可能性用。

### 4.6 为什么内部模型不放 `src/schemas/`？（D6）

`src/schemas/` 是**主图公共契约**目录，里面的东西是所有 Step 共享的接口。
`VisionImage` / `Observation` 只在 Step 3 内部流通（`DiagnosisState` 里没有它们的位置），
所以放在 `vision_agent/vision_models.py`，不占公共契约目录。

### 4.7 为什么回写也要 `model_validate`？

```python
return {"vision": VisionState.model_validate({**state.vision.model_dump(), **updates})}
```

与 Step 2 同理：LangGraph 用返回值**替换整个盒子**（只给部分字段会冲掉其他字段），
而 `model_copy(update=...)` **不跑校验器**，会绕过组长 `StateModel` 的 checkpoint 类型检查。

---

## 5. 测试

| 分组 | 覆盖 |
| :--- | :--- |
| 契约 | `vision` 盒子存在且类型正确；`VisionState` 三字段；旧盒子 `VisionFindings` 不许回来 |
| 工具层（离线） | 魔数判 MIME / data URL / 缩放 / OCR 文本体检（噪声 vs 有用） |
| 编排 | 空图短路（`NO_IMAGE`，零成本）；单张坏图不拖垮整批；**全部失败** → `FAILED` |
| 缺陷映射 | `abnormal` → finding（六字段齐）；**缺四要素 → 跳过 + 进"未核验"** |
| 重试 | 可恢复错误重试（共 3 次）；硬错误（余额不足）立刻抛、不退避 |
| 挂载 | 迷你父图 `START → vision → END`：只回写 `vision`、回流的键不超出公共契约 |
| 真实图 | 跑通一张样例图：`summary` 带 `[图N]` 锚点、`status` 合法、每条 finding 六字段齐 |

> 只有两个**前置条件**缺失才 skip（没有 Key / 没有样例图）；节点建不起来或视觉调用失败
> **必须红**，不许静悄悄变绿。

---

## 6. 已知边界

| # | 边界 | 现状 |
| :---: | :--- | :--- |
| 1 | **HMI / 表计类截图也会产出 finding** | 监控截图里的"告警横幅""趋势卡片"会被模型判成 `abnormal`（`defect_type=OTHER`）。是否收紧口径（只让物理外观缺陷升级为 finding）**暂未定**（用户 2026-09-20 决定先不改） |
| 2 | **视觉结论有抽样波动** | 同一张图两次调用，严重程度偶尔会差一档（如 MODERATE ↔ SEVERE）；断言只校验结构合法性，不绑具体档位 |
| 3 | **OCR 是可选的** | 没装 Tesseract / 没有 `chi_sim` 时，图照样过模型，只是少一层文字辅助；原因会写进"未核验" |
| 4 | **单张图失败只占位、不重跑** | 批量里某张图重试用尽后变成失败占位；整批不会因此中断（这是刻意的） |
| 5 | **裂纹深度、泄漏介质这类信息给不出来** | 单张静态照片的物理极限，会如实写进 `limitations` |

---

## 7. 一句话总结

> **视觉模型负责"看见什么"，代码负责"怎么记账"。**
>
> 一批图 → 逐张（判格式 → OCR 辅助 → 过模型）→ 每个缺陷一条 `VisionFinding`，
> 全文进 `summary`；看过的说 `NO_DEFECT`，没看成的说 `FAILED`，看不清的进"未核验"。
