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
├── tests/                          # 自动化测试用例
│   ├── test_safety_guard.py        # 重点覆盖规则门禁分支与转人工逻辑
│   ├── test_nodes.py               # 单节点逻辑与 Schema 校验测试
│   └── test_pipeline.py            # 端到端 Graph 全链路回归测试
│
├── .env                            # 本地密钥环境变量（严禁入库）
├── .env.example                    # 环境变量模版
├── .gitignore                      # Git 忽略配置
├── main.py                         # 系统执行入口与调试脚本
└── requirements.txt                # 运行依赖项清单
