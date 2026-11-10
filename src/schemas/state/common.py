"""跨 Agent 共享的基础契约。

本模块只放与具体业务步骤无关、但被多处复用的类型：

    StateModel        所有 State 子模型的基类
    StepStatus        各步骤共用的状态别名
    Confidence        统一的置信度词表
    FailureMode       受控故障模式词表
    EvidenceKind      证据类别词表
    EvidenceRef       统一证据引用（kind + id + locator）
    Measurement       带单位与时间窗口的结构化指标

所有枚举都继承 ``str``，既能被 Pydantic 校验，也能直接写入 msgpack checkpoint。
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class StateModel(BaseModel):
    """所有 State 子模型的基类。

    只保留一条硬约束：禁止写入未声明的字段。可序列化性由节点边界的
    checkpoint serde 往返校验负责，不再在模型构造时做手写类型扫描。
    """

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# 状态别名
# =============================================================================

#: 所有步骤都可能出现的通用状态。
#: PENDING=尚未执行，EMPTY=输入缺失，FAILED=执行失败。
StepStatus: TypeAlias = Literal["PENDING", "EMPTY", "FAILED"]

#: Step 2 时序数据状态。
DataStatus: TypeAlias = StepStatus | Literal["OK", "PARTIAL"]

#: 工程师文本状态。
EngineerTextStatus: TypeAlias = StepStatus | Literal["RECORDED"]

#: Step 3 视觉状态。
VisionStatus: TypeAlias = StepStatus | Literal["NO_IMAGE", "NO_DEFECT", "DEFECT_FOUND"]

#: Step 4 手册检索状态。
ManualStatus: TypeAlias = StepStatus | Literal["HIT", "NO_HIT"]


# =============================================================================
# 受控词表
# =============================================================================

class Confidence(str, Enum):
    """统一置信度。

    三处使用点含义一致：**对当前结论的证据把握程度**，不是故障发生概率。
    来源分别记录在字段注释中：

        VisionFinding.confidence  视觉模型自评（score 分档映射）
        Hypothesis.confidence     LLM 对该候选假设的证据强度自评
        FailureModeProposal.confidence
                                  LLM 对首选故障模式的整体证据强度自评
    """

    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class FailureMode(str, Enum):
    """受控故障模式词表。

    LLM 只能从本词表中选择结论，自由文本解释写入 ``summary`` / ``explanation``。
    风险矩阵与安全结论也以本词表为键。
    """

    NORMAL = "NORMAL"
    BEARING_DEGRADATION = "BEARING_DEGRADATION"
    CAVITATION = "CAVITATION"
    SEAL_LEAK = "SEAL_LEAK"
    MOTOR_OVERLOAD = "MOTOR_OVERLOAD"
    HYDRAULIC_IMBALANCE = "HYDRAULIC_IMBALANCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNKNOWN = "UNKNOWN"


class EvidenceKind(str, Enum):
    """证据来源类别。"""

    THRESHOLD = "THRESHOLD"
    MEASUREMENT = "MEASUREMENT"
    VISION = "VISION"
    MANUAL = "MANUAL"
    RULE = "RULE"
    ENGINEER_TEXT = "ENGINEER_TEXT"
    OTHER = "OTHER"


class EvidenceSource(str, Enum):
    """Reasoner 实际使用的输入模态。

    替代原先并存的 ``used_inputs``（小写字符串）与 ``support_sources``
    （大写字符串）：一个字段、一套取值，安全门禁可直接拿它核对真实状态。
    """

    DATA = "DATA"
    VISION = "VISION"
    MANUAL = "MANUAL"
    ENGINEER_TEXT = "ENGINEER_TEXT"


class RiskLevel(str, Enum):
    """风险等级。具体映射由 rules/ram_matrix.json 配置。"""

    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SafetyDecision(str, Enum):
    """安全门禁最终分流决策。只允许由确定性规则代码产生。"""

    PENDING = "PENDING"
    PASS = "PASS"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    BLOCK = "BLOCK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ReviewDecision(str, Enum):
    """人工审核结论。"""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REWORK = "REWORK"


# =============================================================================
# 统一证据引用与结构化指标
# =============================================================================

class EvidenceRef(StateModel):
    """统一证据引用。

    不再用 ``"threshold:VIB_DE_TRIP"`` 这类字符串前缀约定表达来源，
    而是把来源类别、标识和定位信息拆成显式字段。

    kind:
        证据类别，见 :class:`EvidenceKind`。
    id:
        稳定标识，例如规则码、图片 ID、``doc_id#chunk_id``、规则 ID。
    locator:
        进一步定位信息，例如时间窗口、页码、图片区域；没有时留空。
    """

    kind: EvidenceKind
    id: str = Field(min_length=1)
    locator: str = ""


class Measurement(StateModel):
    """一条带单位和时间窗口的结构化指标。

    segment 用于区分窗口级指标（``"overall"``）与分段指标（``"phase:0"``）。
    """

    name: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    segment: str = "overall"
    start_time: str = ""
    end_time: str = ""


def find_measurement(
    measurements: list[Measurement],
    name: str,
    *,
    segment: str = "overall",
) -> float | None:
    """按名称与分段查找指标值；不存在时返回 ``None``。"""

    for measurement in measurements:
        if measurement.name == name and measurement.segment == segment:
            return measurement.value
    return None


__all__ = [
    "Confidence",
    "DataStatus",
    "EngineerTextStatus",
    "EvidenceKind",
    "EvidenceRef",
    "EvidenceSource",
    "FailureMode",
    "ManualStatus",
    "Measurement",
    "ReviewDecision",
    "RiskLevel",
    "SafetyDecision",
    "StateModel",
    "StepStatus",
    "VisionStatus",
    "find_measurement",
]
