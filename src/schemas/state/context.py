"""诊断任务的基础上下文与图片引用。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import StateModel


class ImageRef(StateModel):
    """现场图片引用。图片本体不进入 checkpoint，只保存访问信息。"""

    image_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    source: str = ""
    captured_at: str = ""


class ContextState(StateModel):
    """一次诊断任务的基础上下文。

    trace_id 与 device_id 是流程身份，必须由入口显式传入，且不得为空。
    其余字段可缺失：缺什么就用空值或 ``None`` 表达，不再使用 ``"NONE"`` 哨兵。
    """

    trace_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)

    start_time: str = ""
    end_time: str = ""

    alarm_code: str | None = None
    user_query: str = ""

    image_refs: list[ImageRef] = Field(default_factory=list)
    analysis_mode: Literal["online", "offline"] = "online"
    telemetry_uri: str = ""
    manual_uri: str = ""


__all__ = ["ContextState", "ImageRef"]
