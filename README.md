# CentriPump-PHM 工业故障诊断与排查系统设计文档

本项目是一个结合 **多 Agent 协同（Multi-Agent Workflow）** 与 **确定性工程安全门禁（Deterministic Guardrails）** 的工业运维智能诊断系统。系统基于 LangGraph 进行全局状态机调度，利用大语言模型完成语义提炼、视觉分析与因果假设，并通过纯确定性规则进行风险拦截与工单下发。

## 可复现的离线端到端案例

在项目根目录执行（使用已有依赖环境，无需数据库或 API 密钥）：

```powershell
python scripts/run_demo.py
```

脚本会在现有目录内生成合成轴承温振异常 CSV、演示规程 JSON，构造初始状态后调用
`run_diagnosis()`。主图执行工程师文本、数据子图、无图片状态记录、本地规程检索、
规则归因、安全门禁与交付节点；分析指标和裁决均由本次执行计算，不预填结果。

- 案例输入：`data/sample_inputs/demo_bearing.csv`
- 自制规程：`data/manuals/demo_bearing_manual.json`（仅供演示，按现有 Git 规则忽略；脚本可重建）
- 报告：`docs/demo_diagnosis_report.md`
- 完整状态：`docs/demo_diagnosis_report.json`
- 演示 checkpoint：`data/demo_checkpoints.db`（独立于默认诊断库）

预期结果：361 点、最高温度 85.5℃、最高振动 5.25 mm/s，安全裁决 `BLOCK / CRITICAL`。
六项脚本验收应全部输出 PASS。每次执行创建新的 trace_id；合成采样数据保持一致。
可用 `--output docs/other_report.md` 指定报告路径，旁边同时生成同名 JSON。

此模式显式设置 `context.analysis_mode="offline"`，通过 `telemetry_uri` 指定 CSV、
`manual_uri` 指定本地 JSON 规程。数据描述采用确定性模板；本地规程采用规则码/关键词匹配，
`KEYWORD` 分数是匹配词覆盖率，不是向量相似度。本案例没有图片，不构造视觉结果。
它验证离线模块连接，尚未验证 MySQL、真实 LLM、视觉模型或独立向量 RAG 的在线链路。
原有在线入口和数据库路径保留；不提供上述本地文件路径时沿原有逻辑执行。

调用方也可使用 `run_diagnosis(state, report_path="docs/report.md")` 导出自己的最终状态。
报告明确区分逻辑 `dispatch://` 交付地址和实际报告文件，不执行设备操作或对外派单。
测试可将 `PHM_CHECKPOINT_DB` 指向临时 SQLite 文件，以隔离默认 checkpoint 库。

---

## 1. 目录结构与各模块职责

```text
CentriPump-PHM/
├── configs/                        # 全局与 Agent 级静态配置
│   ├── config.yaml                 # 基础设施配置（模型版本、超时、并发数，支持 ${ENV} 动态注入）
│   └── prompts/                    # 各节点 Prompt 模板解耦目录
│       ├── data_agent.yaml         # Step 2: 时序数据分析 Agent 提示词
│       ├── manual.yaml             # Step 4: 手册条款重写与机理提炼提示词
│       ├── reasoner.yaml           # Step 5: 综合推理与因果归因假设提示词
│       ├── reporter.yaml           # Step 7: 排查工单与溯源证据装配提示词
│       ├── router.yaml             # Step 1: 任务解析、实体抽取与流程路由提示词
│       └── vision_agent.yaml       # Step 3: 视觉特征抽取 Agent 提示词
│
├── data/                           # 本地静态资产与测试样本（通常排除于 Git 外）
│   ├── checkpoints.db              # LangGraph 本地 Checkpoint 持久化数据库
│   ├── manuals/                    # 规程规范、运维操作手册等原始文档
│   │   └── .gitkeep
│   └── sample_inputs/              # 本地测试用的工业时序数据与视觉样本
│       ├── .gitkeep
│       ├── day_2026-09-13_seal_leak.csv # 密封泄漏场景的时序测试数据
│       ├── day_2026-09-14_seal_leak.csv # 密封泄漏场景的时序测试数据
│       ├── PUMP-IS100-80-160-01.csv     # 泵设备 SCADA/时序测试数据
│       └── vision/                  # 视觉 Agent 测试图片样本
│           ├── bad_quality.png       # 低质量图像样本
│           ├── bearing_rust.png      # 轴承锈蚀样本
│           ├── gauge_low.png         # 仪表低读数样本
│           ├── hmi_alarm.png         # HMI 报警信息样本
│           ├── leak_close.png        # 泄漏近景样本
│           ├── leak_wide.png         # 泄漏全景样本
│           ├── nameplate.png         # 设备铭牌样本
│           ├── normal_control.png    # 正常控制面板样本
│           └── thermal_bearing.png   # 轴承热成像样本
│
├── docs/                           # 项目文档与架构图
│   ├── diagnosis_graph.png         # 诊断工作流拓扑图
│   ├── logging.md                  # 日志、Trace 与调用链记录设计说明
│   └── safety_guard.md             # Step 6: 确定性安全门禁与分流机制设计规范
│
├── logs/                           # 运行时日志目录
│   └── app.log                     # 应用运行日志与全链路调用记录
│
├── rules/                          # Step 6: 确定性安全门禁库（核心安全边界，不调用大模型）
│   ├── ram_matrix.json             # RAM 风险矩阵配置（风险等级映射、高危设备黑名单）
│   ├── safety_guard.py             # 确定性安全门禁判定引擎（置信度卡点、拦截分流、动作防护覆盖）
│   ├── thresholds.py               # 规则阈值与安全边界常量（唯一数值出处，GB 50275/企业标准）
│   └── __init__.py
│
├── scripts/                        # 辅助脚本与测试数据生成工具
│   ├── export_csv.py               # 时序数据导出与格式转换脚本
│   ├── run_vision_samples.py       # 批量运行视觉测试样本
│   └── scada_generator.py          # SCADA 模拟数据生成器
│
├── src/                            # 系统核心源码
│   ├── __init__.py
│   │
│   ├── centripump_phm/             # 项目包入口与领域代码包
│   │   └── __init__.py
│   │
│   ├── core/                       # 核心基础能力
│   │   ├── logger.py               # 核心日志能力、Span 生命周期与运行上下文管理
│   │   └── __init__.py
│   │
│   ├── orchestrator/               # 工作流编排层
│   │   ├── graph.py                # LangGraph 状态图定义（节点拓扑、边关系、App 编译、SqliteSaver）
│   │   ├── nodes.py                # 工作流节点适配函数（三要素前置检查、多源证据汇聚、门禁分流与交付）
│   │   ├── runner.py               # 工作流运行入口（创建运行上下文、配置 thread_id/trace_id、执行 Graph）
│   │   └── __init__.py
│   │
│   ├── schemas/                    # Pydantic 强类型数据契约
│   │   ├── report.py               # 最终诊断报告、排查工单与溯源证据结构
│   │   ├── state/                  # LangGraph 状态契约包（按 Agent 拆分）
│   │   │   ├── common.py           # 共享枚举、EvidenceRef、Measurement、StepStatus
│   │   │   ├── context.py          # 诊断上下文与图片引用
│   │   │   ├── engineer_text.py    # 工程师文本状态
│   │   │   ├── data.py             # Step 2 时序分析状态
│   │   │   ├── vision.py           # Step 3 视觉分析状态
│   │   │   ├── manual.py           # Step 4 手册 / RAG 状态
│   │   │   ├── reasoning.py        # Step 5 LLM 故障模式提议
│   │   │   ├── safety.py           # Step 6 不可变安全裁决
│   │   │   ├── human.py            # 追加式人工审核记录
│   │   │   ├── delivery.py         # Step 7 工单交付状态
│   │   │   ├── workflow.py         # 流程元数据
│   │   │   ├── diagnosis.py        # 全局 DiagnosisState 与 create_initial_state
│   │   │   ├── validation.py       # 节点输出白名单与 checkpoint serde 校验
│   │   │   └── __init__.py         # 统一导出
│   │   └── __init__.py
│   │
│   ├── sub_agents/                 # 专项子 Agent 业务实现（独立可测）
│   │   ├── __init__.py
│   │   │
│   │   ├── data_agent/             # Step 2: 时序数据分析专家
│   │   │   ├── analyzer.py         # 时序统计、趋势分析、突变检测与频率特征分析
│   │   │   ├── data_agent_README.md # 数据 Agent 的局部设计与使用说明
│   │   │   ├── data_graph.py       # 数据子 Agent 的 LangGraph 子图定义
│   │   │   ├── data_nodes.py       # 数据子 Agent 节点实现与状态更新
│   │   │   ├── data_tools.py       # 数据 Agent 工具注册与调用封装
│   │   │   ├── repository.py        # 数据仓储与外部/本地数据源访问层
│   │   │   ├── timeseries_tools.py # 数据清洗、窗口处理、FFT 等时序分析工具
│   │   │   └── __init__.py
│   │   │
│   │   ├── manual_agent/           # Step 4: 手册知识与故障机理提炼专家（RAG）
│   │   │   ├── extractor.py        # 手册条款提炼、故障机理抽取与处置要求匹配
│   │   │   ├── retriever.py        # 混合检索（Dense 向量 + BM25 稀疏检索）
│   │   │   └── __init__.py
│   │   │
│   │   └── vision_agent/           # Step 3: 视觉特征抽取专家
│   │       ├── image_io.py         # 图像读取、路径处理与输入输出管理
│   │       ├── ocr_quality.py      # OCR 结果质量评估与有效性判断
│   │       ├── ocr_runner.py       # OCR 模型调用与文字识别流程
│   │       ├── vision_agent_README.md # 视觉 Agent 的局部设计与使用说明
│   │       ├── vision_client.py    # 多模态视觉模型客户端封装
│   │       ├── vision_compose.py   # 多来源视觉结果整理与组合
│   │       ├── vision_models.py    # 视觉模型相关数据结构与模型封装
│   │       ├── vision_nodes.py     # 视觉 Agent 节点实现与状态更新
│   │       └── vision_pipeline.py  # 图像预处理、OCR、视觉模型分析的完整流水线
│   │
│   └── utils/                      # 通用底座基础设施
│       ├── config_loader.py        # 配置加载工具（解析 YAML、读取环境变量、支持 extra='ignore'）
│       ├── database.py             # 数据库连接、会话与持久化访问封装
│       ├── llm_client.py           # LLM 统一客户端（重试、Token 统计、结构化输出等）
│       └── __init__.py
│
├── tests/                          # 自动化测试用例
│   ├── test_logging_context.py     # 日志上下文、trace_id 等调用链测试
│   └── test_safety_guard.py        # Step 6 确定性安全门禁单测与端到端分流测试
│
├── .env.example                    # 环境变量配置模板（真实 .env 不入库）
├── .gitignore                      # Git 忽略规则
├── .python-version                 # Python 版本声明
├── conftest.py                     # pytest 全局配置与测试夹具
├── main.py                         # 系统启动入口与本地调试入口
├── pyproject.toml                  # 项目元数据、依赖与工具配置
├── README.md                       # 项目总体说明与架构文档
├── requirements.txt                # Python 运行依赖清单
├── tree.txt                        # 项目目录结构记录
└── uv.lock                         # uv 依赖版本锁定文件
```

## 2. 流程说明

整个项目的数据交互过程主要由三条主线构成：**LangGraph 工作流、Log 日志记录和数据库数据存取**。三者通过统一的 `trace_id` 进行关联，用于唯一追踪一条完整的故障检修流程。

1. **LangGraph 工作流**负责故障检修流程的编排与执行，管理各 Agent 节点之间的数据传递、状态变化以及中间结果。工作流的 Checkpoint 主要保存流程运行所必需的状态和中间数据，对于图片、完整输出工单、SCADA 时序数据等大型数据块，不直接写入 Checkpoint，而是通过数据库中的数据 ID 进行引用。
2. **Log 日志记录**负责记录工作流运行过程中的关键事件和运行指标，包括节点执行情况、成功或失败、执行耗时、Token 使用量等。日志通过 `trace_id` 与对应的故障检修流程关联，从而支持对一次完整流程的运行过程进行追踪、分析和问题排查。
3. **数据库数据存取**
   负责业务数据及大型数据块的持久化存储，包括故障相关数据、图片、SCADA 时序数据、生成的完整工单及其他流程产物等。工作流中的 Checkpoint 不直接保存这些大型数据，而是保存对应的数据 ID，通过 ID 在需要时读取数据库中的实际数据。数据库中的相关记录同样通过 `trace_id` 与具体故障检修流程建立关联。

因此，三条主线形成了统一的数据链路：

**`trace_id` → 故障检修流程 → LangGraph 工作流 / Log 日志 / 数据库数据**

其中，**LangGraph 负责“流程怎么运行”，Log 负责“流程怎么运行的”，数据库负责“流程产生和使用的数据存在哪里”**，三者共同构成项目完整的数据交互与追踪体系。
