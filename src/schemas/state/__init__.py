"""CentriPump-PHM 的 State 契约包。

按 Agent 拆分文件，本模块统一导出公共契约：

    context.py        诊断上下文
    engineer_text.py  Step 2 工程师文本
    data.py           Step 2 时序分析
    vision.py         Step 3 视觉分析
    manual.py         Step 4 手册 / RAG
    reasoning.py      Step 5 LLM 归因提议
    safety.py         Step 6 确定性裁决
    human.py          人工审核
    delivery.py       Step 7 工单交付
    workflow.py       流程元数据
    diagnosis.py      全局 DiagnosisState 与构造入口
    validation.py     节点输出白名单与 checkpoint serde 校验
"""

from .common import (
    Confidence,
    DataStatus,
    EngineerTextStatus,
    EvidenceKind,
    EvidenceRef,
    EvidenceSource,
    FailureMode,
    ManualStatus,
    Measurement,
    ReviewDecision,
    RiskLevel,
    SafetyDecision,
    StateModel,
    StepStatus,
    VisionStatus,
    find_measurement,
)
from .context import ContextState, ImageRef
from .data import (
    AlarmState,
    DataDescription,
    DataQuality,
    DataSegment,
    DataState,
    DescriptionKind,
    TelemetryRef,
)
from .delivery import DeliveryState
from .diagnosis import DiagnosisState, create_initial_state
from .engineer_text import EngineerTextState
from .human import HumanState, ReviewRecord
from .manual import (
    ManualEvidence,
    ManualState,
    ProcedureRequirement,
    RecommendedAction,
)
from .reasoning import FailureModeProposal, Hypothesis, ReasoningState
from .safety import SafetyState, SafetyVerdict
from .validation import (
    NODE_OUTPUT_WHITELIST,
    assert_serde_roundtrip,
    get_checkpoint_serde,
    node_output,
    validate_node_update,
)
from .vision import VisionFinding, VisionState, confidence_from_score
from .workflow import WorkflowState

__all__ = [
    # 基础
    "AlarmState",
    "Confidence",
    "ContextState",
    "DataDescription",
    "DataQuality",
    "DataSegment",
    "DataState",
    "DataStatus",
    "DeliveryState",
    "DescriptionKind",
    "DiagnosisState",
    "EngineerTextState",
    "EngineerTextStatus",
    "EvidenceKind",
    "EvidenceRef",
    "EvidenceSource",
    "FailureMode",
    "FailureModeProposal",
    "HumanState",
    "Hypothesis",
    "ImageRef",
    "ManualEvidence",
    "ManualState",
    "ManualStatus",
    "Measurement",
    "NODE_OUTPUT_WHITELIST",
    "ProcedureRequirement",
    "ReasoningState",
    "RecommendedAction",
    "ReviewDecision",
    "ReviewRecord",
    "RiskLevel",
    "SafetyDecision",
    "SafetyState",
    "SafetyVerdict",
    "StateModel",
    "StepStatus",
    "TelemetryRef",
    "VisionFinding",
    "VisionState",
    "VisionStatus",
    "WorkflowState",
    "assert_serde_roundtrip",
    "confidence_from_score",
    "create_initial_state",
    "find_measurement",
    "get_checkpoint_serde",
    "node_output",
    "validate_node_update",
]
