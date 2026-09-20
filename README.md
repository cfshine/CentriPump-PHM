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
│       ├── manual.yaml             # Step 4: 手册条款重写与机理提炼提示词
│       ├── reasoner.yaml           # Step 5: 综合推理与因果归因假设提示词
│       └── reporter.yaml           # Step 7: 100% 溯源排查工单装配提示词
│
├── rules/                          # Step 6: 确定性安全门禁库（核心安全边界，绝不调用大模型）
│   ├── ram_matrix.json             # RAM 风险矩阵配置表（风险等级、高危设备黑名单映射）
│   └── safety_guard.py             # 确定性安全门禁判定引擎（置信度卡点、拦截分流、强制转人工）
│
├── src/                            # 系统核心源码
│   ├── schemas/                    # Pydantic 强类型数据契约
│   │   ├── state.py                # LangGraph 全局上下文 DiagnosisState 与各 Agent 出入参模型
│   │   └── report.py               # 最终交付的结构化工单与溯源证据链规范
│   │
│   ├── orchestrator/               # 工作流编排层
│   │   ├── graph.py                # LangGraph 状态图定义（节点拓扑注册、条件分支路由、App 编译）
│   │   └── nodes.py                # 节点适配函数（从 State 解包参数、调用子模块、组装更新回写）
│   │
│   ├── sub_agents/                 # 专项子 Agent 业务实现（独立可测，与 LangGraph 解耦）
│   │   ├── data_agent/             # Step 2: 时序数据分析专家
│   │   │   ├── analyzer.py         # 时序统计指标计算、突变检测、频率分析逻辑
│   │   │   └── timeseries_tools.py # 数据清洗、滑动窗口、FFT 变换辅助工具
│   │   ├── vision_agent/           # Step 3: 视觉特征抽取专家
│   │   │   ├── extractor.py        # 现场图片裂纹、磨损、泄漏缺陷识别
│   │   │   └── image_tools.py      # 图像缩放、增强与多模态模型交互封装
│   │   └── manual_agent/           # Step 4: 手册知识与条款提炼专家 (RAG)
│   │       ├── retriever.py        # 混合检索（Dense 向量 + BM25 稀疏检索）
│   │       └── extractor.py        # 故障机理精炼与规程处置要求匹配
│   │
│   └── utils/                      # 通用底座基础设施
│       ├── llm_client.py           # 模型统一网关（重试机制、Token 统计、结构化解析输出）
│       ├── logger.py               # 全链路调用追踪（基于 trace_id 记录决策链路与审计日志）
│       └── config_loader.py        # 配置加载工具（解析 YAML 并读取 .env 敏感凭据）
│
├── data/                           # 本地静态资产与样本（排除于 Git 外）
│   ├── manuals/                    # 规程规范、运维操作手册原始文档
│   └── sample_inputs/              # 本地测试用的工业时序 CSV、故障图片样本
│
├── .env                            # 本地密钥环境变量（严禁入库）
├── .env.example                    # 环境变量模版
├── .gitignore                      # Git 忽略配置
├── main.py                         # 系统执行入口与调试脚本
└── requirements.txt                # 运行依赖项清单
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
