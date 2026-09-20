# CentriPump-PHM 工业故障诊断与排查系统设计文档

本项目是一个结合 **多 Agent 协同（Multi-Agent Workflow）** 与 **确定性工程安全门禁（Deterministic Guardrails）** 的工业运维智能诊断系统。系统基于 LangGraph 进行全局状态机调度，利用大语言模型完成语义提炼、视觉分析与因果假设，并通过纯确定性规则进行风险拦截与工单下发。

---

## 1. 目录结构与各模块职责

```text
CentriPump-PHM/
├── configs/                        # 全局与 Agent 级静态配置
│   ├── config.yaml                 # 基础设施配置（模型版本、超时、并发数，支持 ${ENV} 动态注入）
│   └── prompts/                    # 各节点 Prompt 模板解耦目录
│       ├── router.yaml             # Step 1: 动态路由与实体抽取提示词
│       ├── data_agent.yaml         # Step 2: 时序数据分析 Agent 提示词
│       ├── vision_agent.yaml       # Step 3: 视觉特征抽取 Agent 提示词
│       ├── manual.yaml             # Step 4: 手册条款重写与机理提炼提示词
│       ├── reasoner.yaml           # Step 5: 综合推理与因果归因假设提示词
│       └── reporter.yaml           # Step 7: 100% 溯源排查工单装配提示词
│
├── rules/                          # Step 6: 确定性安全门禁库（核心安全边界，不调用大模型）
│   ├── __init__.py
│   ├── ram_matrix.json             # RAM 风险矩阵配置表（风险等级、高危设备黑名单映射）
│   ├── safety_guard.py             # 确定性安全门禁判定引擎（置信度卡点、拦截分流、强制转人工）
│   └── thresholds.py               # 规则阈值与安全边界常量
│
├── src/                            # 系统核心源码
│   ├── __init__.py
│   ├── centripump_phm/             # 项目包入口/领域包
│   │   └── __init__.py
│   │
│   ├── schemas/                    # Pydantic 强类型数据契约
│   │   ├── __init__.py
│   │   ├── state.py                # LangGraph 全局上下文 DiagnosisState 与各 Agent 出入参模型
│   │   ├── report.py               # 最终交付的结构化工单与溯源证据链规范
│   │   └── vision.py               # 视觉 Agent 结构化输入/输出契约
│   │
│   ├── orchestrator/               # 工作流编排层
│   │   ├── __init__.py
│   │   ├── graph.py                # LangGraph 状态图定义（节点拓扑注册、条件分支路由、App 编译）
│   │   └── nodes.py                # 节点适配函数（从 State 解包参数、调用子模块、组装更新回写）
│   │
│   ├── sub_agents/                 # 专项子 Agent 业务实现（独立可测，与 LangGraph 解耦）
│   │   ├── __init__.py
│   │   ├── data_agent/             # Step 2: 时序数据分析专家
│   │   │   ├── __init__.py
│   │   │   ├── analyzer.py         # 时序统计指标计算、突变检测、频率分析逻辑
│   │   │   ├── data_graph.py       # 数据子 Agent 的 LangGraph 子图定义
│   │   │   ├── data_nodes.py       # 数据子 Agent 节点适配与状态更新
│   │   │   ├── data_state.py       # 数据子 Agent 局部状态模型
│   │   │   ├── data_tools.py       # 数据子 Agent 工具注册与调用封装
│   │   │   ├── repository.py       # 数据仓储/数据源访问层
│   │   │   ├── timeseries_tools.py # 数据清洗、滑动窗口、FFT 变换辅助工具
│   │   │   └── data_agent_README.md
│   │   │
│   │   ├── vision_agent/           # Step 3: 视觉特征抽取专家
│   │   │   ├── __init__.py
│   │   │   ├── image_tools.py      # 图像缩放、增强与多模态模型交互封装
│   │   │   ├── vision_graph.py     # 视觉子 Agent 的 LangGraph 子图定义
│   │   │   ├── vision_nodes.py     # 视觉子 Agent 节点适配与状态更新
│   │   │   ├── vision_state.py     # 视觉子 Agent 局部状态模型
│   │   │   └── vision_agent_README.md
│   │   │
│   │   └── manual_agent/           # Step 4: 手册知识与条款提炼专家 (RAG)
│   │       ├── __init__.py
│   │       ├── retriever.py        # 混合检索（Dense 向量 + BM25 稀疏检索）
│   │       └── extractor.py        # 故障机理精炼与规程处置要求匹配
│   │
│   └── utils/                      # 通用底座基础设施
│       ├── __init__.py
│       ├── llm_client.py           # 模型统一网关（重试机制、Token 统计、结构化解析输出）
│       ├── logger.py               # 全链路调用追踪（基于 trace_id 记录决策链路与审计日志）
│       ├── config_loader.py        # 配置加载工具（解析 YAML 并读取 .env 敏感凭据）
│       └── database.py             # 数据库连接、持久化与查询封装
│
├── data/                           # 本地静态资产与样本（排除于 Git 外）
│   ├── manuals/                    # 规程规范、运维操作手册原始文档
│   │   └── .gitkeep
│   └── sample_inputs/              # 本地测试用的工业时序 CSV、故障图片样本
│       ├── .gitkeep
│       ├── day_2026-09-13_seal_leak.csv
│       ├── day_2026-09-14_seal_leak.csv
│       ├── PUMP-IS100-80-160-01.csv
│       └── vision/
│           ├── bad_quality.png
│           ├── bearing_rust.png
│           ├── gauge_low.png
│           ├── hmi_alarm.png
│           ├── leak_close.png
│           ├── leak_wide.png
│           ├── nameplate.png
│           ├── normal_control.png
│           └── thermal_bearing.png
│
├── docs/                           # 项目文档与架构图
│   └── diagnosis_graph.png
│
├── scripts/                        # 辅助脚本与数据生成工具
│   ├── export_csv.py               # 导出 CSV 数据
│   ├── run_vision_samples.py       # 批量运行视觉样本
│   └── scada_generator.py          # SCADA 模拟数据生成器
│
├── tests/                          # 测试用例
│   ├── test_nodes.py
│   └── test_vision_agent.py
│
├── .env.example                    # 环境变量模版（真实 .env 不入库）
├── .gitignore                      # Git 忽略配置
├── .python-version                 # Python 版本声明
├── conftest.py                     # pytest 全局配置/夹具
├── main.py                         # 系统执行入口与调试脚本
├── pyproject.toml                  # 项目元数据、构建与工具配置
├── README.md                       # 项目说明
├── requirements.txt                # 运行依赖项清单
├── tree.txt                        # 目录结构记录
└── uv.lock                         # uv 依赖锁文件
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
