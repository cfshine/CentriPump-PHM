"""工程师文本输入状态。"""

from __future__ import annotations

from .common import EngineerTextStatus, StateModel


class EngineerTextState(StateModel):
    """工程师现场描述的处理结果。"""

    status: EngineerTextStatus = "PENDING"
    text: str = ""
    summary: str = ""


__all__ = ["EngineerTextState"]
