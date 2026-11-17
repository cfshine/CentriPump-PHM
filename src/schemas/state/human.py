"""人工审核状态。

审核记录采用追加式列表：二审、三审不会覆盖一审，完整保留审核轨迹。
"""

from __future__ import annotations

from pydantic import Field

from .common import ReviewDecision, StateModel


class ReviewRecord(StateModel):
    """一次人工审核。"""

    decision: ReviewDecision
    reviewer: str = Field(min_length=1)
    reviewed_at: str = Field(min_length=1)
    comment: str = ""


class HumanState(StateModel):
    """人工介入信息。

    接入 LangGraph 时，人工审核节点可用 ``langgraph.types.interrupt()`` 挂起图，
    恢复后把本次审核以追加 ``ReviewRecord`` 的方式写回；历史记录不覆盖。
    """

    reviews: list[ReviewRecord] = Field(default_factory=list)


__all__ = ["HumanState", "ReviewRecord"]
