"""Step 7 最终交付状态。"""

from __future__ import annotations

from pydantic import Field

from .common import EvidenceRef, StateModel


class DeliveryState(StateModel):
    """工单交付元数据。

    完整工单正文不进入 checkpoint；这里只保存工单身份、内容哈希与证据链。
    """

    diagnosed_at: str = ""
    report_sha256: str = ""
    report_uri: str = ""
    traceability: list[EvidenceRef] = Field(default_factory=list)


__all__ = ["DeliveryState"]
