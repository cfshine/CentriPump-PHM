"""流程级元数据。"""

from __future__ import annotations

from pydantic import Field

from .common import StateModel


class WorkflowState(StateModel):
    """与具体业务 Agent 无关的流程信息。"""

    schema_version: str = "2.0"
    degraded_steps: list[str] = Field(default_factory=list)


__all__ = ["WorkflowState"]
