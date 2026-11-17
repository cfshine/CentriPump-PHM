"""LangGraph 全局状态与其唯一构造入口。"""

from __future__ import annotations

from pydantic import Field

from .common import StateModel
from .context import ContextState, ImageRef
from .data import DataState
from .delivery import DeliveryState
from .engineer_text import EngineerTextState
from .human import HumanState
from .manual import ManualState
from .reasoning import ReasoningState
from .safety import SafetyState
from .vision import VisionState
from .workflow import WorkflowState


class DiagnosisState(StateModel):
    """CentriPump-PHM 的 LangGraph 全局状态。

    九个业务盒子与各自的生产者一一对应：

        context      入口
        engineer_text Step 2 工程师文本
        data         Step 2 时序分析
        vision       Step 3 视觉分析
        manual       Step 4 手册 / RAG
        reasoning    Step 5 LLM 归因提议
        safety       Step 6 确定性裁决
        human        人工审核
        delivery     Step 7 工单交付
        workflow     流程元数据
    """

    context: ContextState

    engineer_text: EngineerTextState = Field(default_factory=EngineerTextState)
    data: DataState = Field(default_factory=DataState)
    vision: VisionState = Field(default_factory=VisionState)
    manual: ManualState = Field(default_factory=ManualState)
    reasoning: ReasoningState = Field(default_factory=ReasoningState)
    safety: SafetyState = Field(default_factory=SafetyState)
    human: HumanState = Field(default_factory=HumanState)
    delivery: DeliveryState = Field(default_factory=DeliveryState)
    workflow: WorkflowState = Field(default_factory=WorkflowState)


def create_initial_state(
    *,
    trace_id: str,
    device_id: str,
    start_time: str = "",
    end_time: str = "",
    user_query: str = "",
    alarm_code: str | None = None,
    image_refs: list[ImageRef] | None = None,
) -> DiagnosisState:
    """创建一次诊断流程的初始状态。

    ``trace_id`` 与 ``device_id`` 是必填身份，由入口显式生成 / 传入；
    其余盒子使用模型默认值，不再在这里重复填一遍空值。
    """

    return DiagnosisState(
        context=ContextState(
            trace_id=trace_id,
            device_id=device_id,
            start_time=start_time,
            end_time=end_time,
            user_query=user_query,
            alarm_code=alarm_code,
            image_refs=list(image_refs or []),
        )
    )


__all__ = ["DiagnosisState", "create_initial_state"]
