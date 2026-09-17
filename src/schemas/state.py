from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypedDict


# ============================================================
# 1. 基础类型
# ============================================================

# 当前工作流整体状态。
# 注意：
# 这不是日志状态，也不是数据库中的业务状态。
# 它描述的是“当前这一次 LangGraph 执行到什么程度、手里有什么材料”。
WorkflowStatus = Literal[
    "pending",       # 尚未开始
    "running",       # 执行中
    "completed",     # 正常完成
    "failed",        # 执行失败
    "blocked",       # 被安全门禁拦截
]


# ============================================================
# 2. Artifact：大型原始材料的引用
# ============================================================

class ArtifactRef(TypedDict):
    """
    一个大型外部材料的引用。

    图片、PDF、SCADA 原始数据、最终报告等，不直接放进 LangGraph State。
    State / Checkpoint 中只保存这个引用。

    实际文件可以放在：
        - MinIO
        - S3
        - OSS
        - 本地文件存储

    DB 中也可以保存对应的 artifact 记录。
    """

    # 全局唯一的材料 ID。
    # 例如：
    #   IMG-001
    #   SCADA-001
    #   MANUAL-001
    #   REPORT-001
    artifact_id: str

    # 材料类型。
    artifact_type: Literal[
        "image",
        "scada",
        "manual",
        "report",
        "document",
        "video",
        "other",
    ]

    # 原始文件名称。
    # 例如：
    #   pump_003.jpg
    #   pump_003_scada.parquet
    #   centrifugal_pump_manual.pdf
    name: str

    # 文件 MIME 类型。
    # 例如：
    #   image/jpeg
    #   application/pdf
    #   application/octet-stream
    mime_type: str

    # 文件在对象存储 / 文件系统中的 URI。
    #
    # 注意：
    # 这里只是引用，不是文件内容。
    uri: str

    # 文件大小，单位 bytes。
    size: int

    # 文件 SHA256。
    #
    # 用于确认文件完整性，以及避免同一文件重复上传。
    sha256: str


# ============================================================
# 3. 工程师 / 用户提供的原始文本
# ============================================================

class EngineerInput(TypedDict):
    """
    工程师或者运维人员提供的文字描述。

    这类数据通常很小，可以直接进入 State。
    """

    # 原始输入文本。
    text: str

    # 输入来源。
    source: Literal[
        "engineer",
        "operator",
        "user",
        "system",
    ]

    # 输入时间。
    created_at: datetime


# ============================================================
# 4. Router 输出
# ============================================================

class RouterResult(TypedDict):
    """
    Router 对当前诊断任务进行路由后的结果。

    Router 本身只负责决定需要哪些分析路径。
    """

    # 诊断任务类型。
    intent: Literal[
        "equipment_fault",
        "performance_anomaly",
        "maintenance",
        "alarm_analysis",
        "other",
    ]

    # 是否需要获取 SCADA 数据。
    need_scada: bool

    # 是否需要工程师文本分析。
    need_engineer_text: bool

    # 是否需要视觉分析。
    need_vision: bool

    # Router 给出的简短任务描述。
    task_summary: str


# ============================================================
# 5. SCADA 数据
# ============================================================

class ScadaDataRef(TypedDict):
    """
    SCADA 原始数据的引用。

    原始 SCADA 数据可能非常大，因此不直接进入 Checkpoint。
    """

    # 对应 Artifact 的 ID。
    artifact_id: str

    # 设备 ID。
    equipment_id: str

    # SCADA 数据开始时间。
    start_time: datetime

    # SCADA 数据结束时间。
    end_time: datetime

    # 数据包含的测点。
    #
    # 例如：
    # [
    #     "temperature",
    #     "pressure",
    #     "flow",
    #     "vibration"
    # ]
    metrics: list[str]


class ScadaAnomaly(TypedDict):
    """
    SCADA Agent 找到的一处异常。
    """

    # 测点名称。
    metric: str

    # 异常发生时间。
    timestamp: datetime

    # 异常值。
    value: float

    # 正常范围。
    normal_range: tuple[float, float]

    # 异常类型。
    anomaly_type: Literal[
        "high",
        "low",
        "fluctuation",
        "trend",
        "spike",
        "drop",
        "other",
    ]

    # 简短描述。
    description: str


class ScadaResult(TypedDict):
    """
    SCADA Agent 的结构化分析结果。

    注意：
    这里只保存分析结果，不保存完整 SCADA 原始数据。
    """

    # SCADA 原始数据引用。
    data_ref: ScadaDataRef

    # 检测出的异常。
    anomalies: list[ScadaAnomaly]

    # 总体趋势描述。
    trend_summary: str

    # SCADA Agent 对当前数据的摘要。
    summary: str


# ============================================================
# 6. Vision 视觉分析
# ============================================================

class VisionObservation(TypedDict):
    """
    Vision Agent 从图片中观察到的一个现象。

    注意：
    这里应该描述“观察到什么”，而不是直接下最终故障结论。
    """

    # 对象位置。
    # 例如：
    #   "泵轴承区域"
    #   "机械密封区域"
    location: str

    # 观察结果。
    observation: str

    # 视觉证据的严重程度。
    severity: Literal[
        "normal",
        "minor",
        "moderate",
        "severe",
        "unknown",
    ]


class VisionResult(TypedDict):
    """
    Vision Agent 的输出。
    """

    # 本次视觉分析使用的图片。
    image_refs: list[ArtifactRef]

    # 图片中的观察结果。
    observations: list[VisionObservation]

    # 视觉分析摘要。
    summary: str


# ============================================================
# 7. 工程师文本分析
# ============================================================

class EngineerAnalysisResult(TypedDict):
    """
    对工程师原始描述进行语义提炼后的结果。

    例如：
        “3号泵这几天声音越来越大，而且振动明显”
    
    可以提炼为：
        symptom = ["abnormal noise", "increased vibration"]
    """

    # 设备当前表现出的症状。
    symptoms: list[str]

    # 发生时间 / 持续时间。
    duration: str

    # 发生条件。
    operating_condition: str

    # 工程师提供的其他重要上下文。
    context: list[str]

    # 对工程师描述的结构化摘要。
    summary: str


# ============================================================
# 8. Evidence Merge
# ============================================================

class EvidenceItem(TypedDict):
    """
    合并后的单条证据。

    Evidence Merge 的作用是把：

        工程师文本
        SCADA
        Vision
        其他材料

    统一整理成 Reasoner 可以使用的证据结构。
    """

    # 证据唯一 ID。
    evidence_id: str

    # 证据来源。
    source: Literal[
        "engineer",
        "scada",
        "vision",
        "manual",
        "other",
    ]

    # 证据内容。
    #
    # 这里应该是结构化的小型结果，而不是原始大文件。
    content: str

    # 证据对应的 Artifact。
    #
    # 例如：
    # 某个 SCADA 异常可以关联 SCADA-001。
    artifact_ids: list[str]

    # 证据的重要程度。
    relevance: Literal[
        "low",
        "medium",
        "high",
    ]


class EvidenceMergeResult(TypedDict):
    """
    Evidence Merge 的完整结果。
    """

    # 所有可供后续 Reasoner 使用的证据。
    evidence: list[EvidenceItem]

    # 对全部证据进行的总体摘要。
    summary: str


# ============================================================
# 9. Manual：设备手册 / 工程知识匹配结果
# ============================================================

class ManualReference(TypedDict):
    """
    从设备手册中找到的相关内容。

    注意：
    手册 PDF 本身是 Artifact。
    这里保存的是被检索出来的相关知识。
    """

    # 手册 Artifact。
    artifact_id: str

    # 手册中的章节。
    section: str

    # 相关内容。
    content: str

    # 与当前故障的相关性。
    relevance: Literal[
        "low",
        "medium",
        "high",
    ]


class ManualResult(TypedDict):
    """
    Manual Agent 的输出。
    """

    # 找到的相关手册内容。
    references: list[ManualReference]

    # 手册知识总结。
    summary: str


# ============================================================
# 10. Reasoner：故障假设
# ============================================================

class FaultHypothesis(TypedDict):
    """
    Reasoner 提出的一个故障假设。

    注意：
    这里是“假设”，不是最终确定事实。

    例如：
        bearing_degradation
        seal_leakage
        cavitation
    """

    # 故障类型。
    fault_type: str

    # 故障描述。
    description: str

    # 支持该假设的证据 ID。
    supporting_evidence_ids: list[str]

    # 与该假设冲突的证据 ID。
    conflicting_evidence_ids: list[str]

    # 模型对该假设的置信度。
    confidence: float


class ReasoningResult(TypedDict):
    """
    Reasoner 的完整推理结果。
    """

    # 模型提出的所有故障假设。
    hypotheses: list[FaultHypothesis]

    # 模型认为当前最可能的故障。
    #
    # 注意：
    # 这仍然只是 LLM 的判断。
    # 最终是否允许采取行动，需要经过 safety_guard。
    primary_hypothesis: str

    # Reasoner 的推理摘要。
    summary: str

    # 模型输出的建议。
    recommendation: str


# ============================================================
# 11. Safety Guard：确定性安全门禁
# ============================================================

class SafetyRuleResult(TypedDict):
    """
    单条确定性安全规则的检查结果。

    这个结构非常重要，因为以后需要追溯：
        “为什么这个操作被允许 / 拦截？”
    """

    # 规则 ID。
    # 例如：
    #   RAM-001
    #   SAFETY-003
    rule_id: str

    # 规则版本。
    rule_version: str

    # 是否通过。
    passed: bool

    # 规则检查结果说明。
    message: str


class SafetyGuardResult(TypedDict):
    """
    Safety Guard 的完整结果。

    Safety Guard 不使用 LLM 做最终安全决策。
    """

    # 是否通过全部安全检查。
    passed: bool

    # 最终风险等级。
    risk_level: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ]

    # 每一条规则的检查结果。
    rules: list[SafetyRuleResult]

    # 如果被拦截，这里说明原因。
    block_reason: str

    # 安全门禁最终允许执行的动作。
    #
    # 例如：
    #   "generate_report"
    #   "create_work_order"
    #   "manual_inspection"
    #   "shutdown_equipment"
    allowed_actions: list[str]


# ============================================================
# 12. Reporter：最终报告
# ============================================================

class ReportResult(TypedDict):
    """
    Reporter 的最终结果。

    最终报告文件本身不进入 State。
    State 中只保存 ArtifactRef。
    """

    # 最终报告的 Artifact。
    report_artifact: ArtifactRef

    # 报告标题。
    title: str

    # 最终诊断结论。
    conclusion: str

    # 最终建议。
    recommendation: str


# ============================================================
# 13. 最终业务诊断结果
# ============================================================

class DiagnosisResult(TypedDict):
    """
    一次诊断最终形成的结构化业务结果。

    这个结果未来会同步到 Business DB。
    """

    # 最终故障类型。
    fault_type: str

    # 最终风险等级。
    risk_level: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ]

    # 最终诊断结论。
    conclusion: str

    # 最终处理建议。
    recommendation: str

    # 最终使用的证据。
    evidence_ids: list[str]


# ============================================================
# 14. Workflow State
# ============================================================

class DiagnosisState(TypedDict):
    """
    CentriPump-PHM LangGraph 的完整工作流状态。

    ============================================================
    设计原则
    ============================================================

    这个 State 会被 LangGraph Checkpointer 持久化。

    因此：

    1. 小型结构化数据可以直接保存。
    2. 大型文件只保存 ArtifactRef。
    3. Log 不放进 State。
    4. 数据库连接、Session 等运行时对象不放进 State。
    5. HTTP Request、LLM Client 等对象不放进 State。
    6. 最终业务数据可以从 State 同步到 Business DB。
    """

    # --------------------------------------------------------
    # A. Workflow / Run 基础信息
    # --------------------------------------------------------

    # 一次诊断案件的业务 ID。
    #
    # 对应数据库中的 diagnosis_case。
    case_id: str

    # 一次具体的 LangGraph 执行 ID。
    #
    # 一个 case 理论上可以有多次 run。
    #
    # 例如：
    #   第一次运行失败
    #   第二次从 checkpoint 恢复
    #
    # 两次可以属于同一个 case。
    run_id: str

    # 当前 Graph 版本。
    #
    # 用于以后追溯：
    # “这个诊断当时运行的是哪一个工作流版本？”
    graph_version: str

    # 工作流当前状态。
    workflow_status: WorkflowStatus

    # 工作流开始时间。
    started_at: datetime

    # 最近一次 State 更新时间。
    updated_at: datetime

    # --------------------------------------------------------
    # B. 原始输入材料
    # --------------------------------------------------------

    # 工程师 / 操作员输入的文本。
    engineer_input: EngineerInput

    # 当前案件涉及的设备 ID。
    equipment_id: str

    # 当前案件所有原始材料。
    #
    # 图片、SCADA、PDF 等大型内容全部通过 ArtifactRef 引用。
    artifacts: list[ArtifactRef]

    # --------------------------------------------------------
    # C. Router
    # --------------------------------------------------------

    # Router 的分析结果。
    router: RouterResult

    # --------------------------------------------------------
    # D. SCADA Agent
    # --------------------------------------------------------

    # SCADA 原始数据引用。
    #
    # 这里不是完整 SCADA 数据。
    scada_data_ref: ScadaDataRef | None

    # SCADA 分析结果。
    scada_result: ScadaResult | None

    # --------------------------------------------------------
    # E. Engineer Text Agent
    # --------------------------------------------------------

    # 工程师文本语义分析结果。
    engineer_analysis: EngineerAnalysisResult | None

    # --------------------------------------------------------
    # F. Vision Agent
    # --------------------------------------------------------

    # Vision 分析结果。
    vision_result: VisionResult | None

    # --------------------------------------------------------
    # G. Evidence Merge
    # --------------------------------------------------------

    # 合并后的证据。
    evidence: EvidenceMergeResult | None

    # --------------------------------------------------------
    # H. Manual
    # --------------------------------------------------------

    # 从设备手册中找到的相关知识。
    manual: ManualResult | None

    # --------------------------------------------------------
    # I. Reasoner
    # --------------------------------------------------------

    # LLM 故障推理结果。
    reasoning: ReasoningResult | None

    # --------------------------------------------------------
    # J. Safety Guard
    # --------------------------------------------------------

    # 确定性安全门禁结果。
    safety_guard: SafetyGuardResult | None

    # --------------------------------------------------------
    # K. Final Diagnosis
    # --------------------------------------------------------

    # 最终结构化诊断结果。
    #
    # 通常在 safety_guard 通过之后形成。
    diagnosis: DiagnosisResult | None

    # --------------------------------------------------------
    # L. Reporter
    # --------------------------------------------------------

    # 最终报告结果。
    report: ReportResult | None

    # --------------------------------------------------------
    # M. 工单
    # --------------------------------------------------------

    # 是否需要创建工单。
    #
    # 这个字段最终应该由 Safety Guard 决定，而不是 LLM 自己决定。
    work_order_required: bool

    # 创建后的工单 ID。
    #
    # 工单真正的数据应该放 Business DB。
    # State 中只保存 ID，方便后续节点引用。
    work_order_id: str | None