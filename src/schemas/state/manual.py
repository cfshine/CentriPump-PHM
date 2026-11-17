"""Step 4 手册 / RAG 状态。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import EvidenceRef, ManualStatus, StateModel


class ManualEvidence(StateModel):
    """一条手册证据。只保存身份、短摘录与检索元数据，不保存整页正文。"""

    doc_id: str = ""
    chunk_id: str = ""
    section: str = ""
    page: int = 0
    quote: str = ""
    score: float = 0.0
    channel: Literal["DENSE", "BM25", "HYBRID", "KEYWORD"] = "HYBRID"


class ProcedureRequirement(StateModel):
    """手册中的规程要求。"""

    requirement: str = ""
    source_ref: EvidenceRef


class RecommendedAction(StateModel):
    """基于证据形成的建议动作。

    建议动作不等于安全放行；是否允许执行仍由 Safety Guard 裁决。
    """

    action: str = ""
    priority: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    source_ref: EvidenceRef


class ManualState(StateModel):
    """Step 4 Manual Agent 输出。"""

    status: ManualStatus = "PENDING"
    evidences: list[ManualEvidence] = Field(default_factory=list)
    mechanism: str = ""
    requirements: list[ProcedureRequirement] = Field(default_factory=list)
    actions: list[RecommendedAction] = Field(default_factory=list)


__all__ = [
    "ManualEvidence",
    "ManualState",
    "ProcedureRequirement",
    "RecommendedAction",
]
