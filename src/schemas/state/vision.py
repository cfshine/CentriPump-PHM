"""Step 3 现场图片视觉分析状态。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import Confidence, StateModel, VisionStatus


class VisionFinding(StateModel):
    """单条视觉缺陷发现。每条都必须能回溯到具体 image_id。"""

    image_id: str = ""
    defect_type: str = ""
    severity: Literal["MINOR", "MODERATE", "SEVERE"] = "MINOR"
    location: str = ""

    confidence: Confidence = Confidence.UNKNOWN
    """视觉模型对自身判断的把握程度。

    来源：``vision_models.Observation.confidence``（0~1）按
    :func:`confidence_from_score` 分档映射。不是故障概率。
    """

    evidence: str = ""


def confidence_from_score(score: float) -> Confidence:
    """把视觉模型的 0~1 自评分映射到统一 Confidence 词表。"""

    if score >= 0.8:
        return Confidence.HIGH
    if score >= 0.5:
        return Confidence.MEDIUM
    if score > 0.0:
        return Confidence.LOW
    return Confidence.UNKNOWN


class VisionState(StateModel):
    """Step 3 Vision Agent 输出。"""

    status: VisionStatus = "PENDING"
    findings: list[VisionFinding] = Field(default_factory=list)
    summary: str = ""


__all__ = ["VisionFinding", "VisionState", "confidence_from_score"]
