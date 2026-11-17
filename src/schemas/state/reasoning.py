"""Step 5 故障归因的 LLM 提议状态。

``reasoning`` 只保存 LLM 的**提议**。确定性结论由 Step 6 Safety Guard 产生，
并写入 ``safety.verdict``；Reporter 只能使用 verdict 中已确认的内容。
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import (
    Confidence,
    EvidenceRef,
    EvidenceSource,
    FailureMode,
    StateModel,
)


class Hypothesis(StateModel):
    """一条候选故障模式假设，包括被否定或证据不足的假设。"""

    failure_mode: FailureMode
    confidence: Confidence = Confidence.UNKNOWN
    """LLM 对该候选假设的证据强度自评。来源：Reasoner 提议。"""

    support_refs: list[EvidenceRef] = Field(default_factory=list)
    contradicting_refs: list[EvidenceRef] = Field(default_factory=list)
    status: Literal["SUPPORTED", "INSUFFICIENT", "CONFLICTED"] = "INSUFFICIENT"
    explanation: str = ""


class FailureModeProposal(StateModel):
    """LLM 给出的首选故障模式提议。"""

    failure_mode: FailureMode
    confidence: Confidence = Confidence.UNKNOWN
    """LLM 对首选故障模式的整体证据强度自评。来源：Reasoner 提议。"""

    support_refs: list[EvidenceRef] = Field(default_factory=list)
    explanation: str = ""


class ReasoningState(StateModel):
    """Step 5 Reasoner 输出：只含提议，不含最终安全结论。"""

    proposal: FailureModeProposal | None = None
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    """Reasoner 实际使用的输入模态；用于安全门禁核对输入幻觉。"""

    summary: str = ""
    insufficient_evidence: bool = False
    conflicts: list[str] = Field(default_factory=list)

    @property
    def failure_mode(self) -> FailureMode | None:
        """首选提议的故障模式；没有提议时返回 ``None``。"""

        return self.proposal.failure_mode if self.proposal else None

    @property
    def confidence(self) -> Confidence:
        """首选提议的置信度；没有提议时返回 ``UNKNOWN``。"""

        return self.proposal.confidence if self.proposal else Confidence.UNKNOWN


__all__ = ["FailureModeProposal", "Hypothesis", "ReasoningState"]
