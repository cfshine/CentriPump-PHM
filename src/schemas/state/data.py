"""Step 2 时序数据分析状态。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import DataStatus, Measurement, StateModel


class TelemetryRef(StateModel):
    """原始遥测数据引用。原始 SCADA 数据不进入 checkpoint。"""

    source: str = ""
    trace_id: str = ""
    device_id: str = ""
    start_time: str = ""
    end_time: str = ""


class DataQuality(StateModel):
    """时序数据质量。"""

    status: DataStatus = "PENDING"
    total_points: int = 0
    reason: str = ""


class AlarmState(StateModel):
    """报警码相关信息。

    缺失即为空列表，不再用 ``"NONE"`` 字符串作哨兵。
    """

    effective: list[str] = Field(default_factory=list)
    last: list[str] = Field(default_factory=list)
    all: list[str] = Field(default_factory=list)


DescriptionKind = Literal[
    "TREND",
    "ANOMALY",
    "THRESHOLD",
    "CORRELATION",
    "ALARM",
    "OTHER",
]


class DataDescription(StateModel):
    """对数据现象的语义化描述。只描述现象，不做故障归因。"""

    kind: DescriptionKind
    description: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)


class DataSegment(StateModel):
    """一个连续运行阶段的边界与规则命中。"""

    index: int
    state: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_sec: int = 0
    data_points: int = 0
    rule_hits: list[str] = Field(default_factory=list)
    ramp_start_time: str | None = None


class DataState(StateModel):
    """Step 2 Data Agent 的全部输出。

    指标统一存为 ``Measurement``：每个数值都带单位、时间窗口和所属分段，
    不再出现 ``{"rms": 12.3}`` 这类无法判断量纲的裸字典。
    """

    telemetry_ref: TelemetryRef = Field(default_factory=TelemetryRef)
    quality: DataQuality = Field(default_factory=DataQuality)

    measurements: list[Measurement] = Field(default_factory=list)
    segments: list[DataSegment] = Field(default_factory=list)

    threshold_flags: list[str] = Field(default_factory=list)
    alarms: AlarmState = Field(default_factory=AlarmState)
    descriptions: list[DataDescription] = Field(default_factory=list)


__all__ = [
    "AlarmState",
    "DataDescription",
    "DataQuality",
    "DataSegment",
    "DataState",
    "DescriptionKind",
    "TelemetryRef",
]
