"""Step 6 确定性安全门禁结论。

安全结论只允许由 ``rules/safety_guard.py`` 产生。结论对象一旦构造即冻结，
上游 LLM 或下游 Reporter 都不能改写它的字段。
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from .common import FailureMode, RiskLevel, SafetyDecision, StateModel


class SafetyVerdict(StateModel):
    """不可变的安全裁决。

    同一份输入 + 同一规则版本必须重放出完全相同的 verdict；
    ``input_hash`` 与 ``rule_version`` 就是这条可审计承诺的凭据。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: SafetyDecision
    risk_level: RiskLevel
    failure_mode: FailureMode
    """经门禁确认后的最终故障模式；证据不足时为 INSUFFICIENT_EVIDENCE。"""

    rule_hits: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    rule_version: str = Field(min_length=1)
    input_hash: str = Field(min_length=1)


class SafetyState(StateModel):
    """State 中的安全盒子：包住不可变 verdict，并提供只读访问属性。"""

    verdict: SafetyVerdict | None = None

    @property
    def decision(self) -> SafetyDecision:
        return self.verdict.decision if self.verdict else SafetyDecision.PENDING

    @property
    def risk_level(self) -> RiskLevel:
        return self.verdict.risk_level if self.verdict else RiskLevel.UNKNOWN

    @property
    def failure_mode(self) -> FailureMode:
        return self.verdict.failure_mode if self.verdict else FailureMode.UNKNOWN

    @property
    def rule_hits(self) -> list[str]:
        return self.verdict.rule_hits if self.verdict else []

    @property
    def reasons(self) -> list[str]:
        return self.verdict.reasons if self.verdict else []

    @property
    def rule_version(self) -> str:
        return self.verdict.rule_version if self.verdict else ""

    @property
    def input_hash(self) -> str:
        return self.verdict.input_hash if self.verdict else ""


__all__ = ["SafetyState", "SafetyVerdict"]
