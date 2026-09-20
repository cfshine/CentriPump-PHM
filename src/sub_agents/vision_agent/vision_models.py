"""Step 3 视觉环节的**内部**数据模型（★ 不再是公共契约的一部分）。

  ``VisionFinding``（每个缺陷一条）。

分层（依赖单向，从上往下）：
    vision_nodes → vision_pipeline → vision_client → vision_compose → 工具层
    本文件是最底层的纯数据模型：谁都可以 import 它，它只 import pydantic。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: 单个观测的极性（封闭词表）：
#: abnormal=可见异常 / normal=可见正常 / unknown=看不出或无法判断
Polarity = Literal["abnormal", "normal", "unknown"]

#: 缺陷严重程度（封闭词表，与``VisionFinding.severity`` 对齐）：
#: MINOR=轻微、局部、不影响运行 / MODERATE=明显但不危及运行 / SEVERE=严重或危及运行
Severity = Literal["MINOR", "MODERATE", "SEVERE"]


class Observation(BaseModel):
    """一条"看见了什么"。只描述现象，不写故障原因。

    基础字段（三种极性都要填）：
        target:   画面中的对象，如 "驱动端轴承箱" / "出口压力表" / "顶部告警条"。
        finding:  看见了什么，只描述不归因（如 "压盖下缘有深色液体连续流淌"）。
        polarity: 极性，abnormal / normal / unknown（封闭词表，供下游写 if）。

    缺陷字段（**只有 ``polarity == "abnormal"`` 时有意义**，提示词要求模型必填；
    节点会据此把该观测升级成组长的 ``VisionFinding``）：
        defect_type: 缺陷类型，建议用 LEAK / WEAR / CRACK / SCALE / LOOSE 这类词。
        severity:    严重程度 MINOR / MODERATE / SEVERE（判定口径见提示词）。
        confidence:  模型**对自己这个判断的把握**（0~1）。
                     ★ 它不是故障概率，也不等价于整条诊断链路的最终置信度。
        evidence:    判断依据 —— 画面上看见了什么支撑这个结论；只写画面依据，不写原因。
    """

    target: str = Field(..., description="画面中的对象，如驱动端轴承箱、出口压力表、铭牌")
    finding: str = Field(..., description="看见了什么，只描述不归因")
    polarity: Polarity = "unknown"

    defect_type: str = Field(
        default="",
        description="缺陷类型（polarity=abnormal 时必填），如 LEAK / WEAR / CRACK / SCALE",
    )
    severity: Severity | None = Field(
        default=None,
        description="严重程度（polarity=abnormal 时必填）：MINOR / MODERATE / SEVERE",
    )
    confidence: float = Field(
        default=0.0,
        description="你对该判断的把握，0~1（不是故障概率，也不是最终诊断置信度）",
    )
    evidence: str = Field(
        default="",
        description="判断依据：画面上看到了什么支撑这个结论（polarity=abnormal 时必填）",
    )


class VisionImage(BaseModel):
    """**一张图**的结论。

    字段：
        natural_description: 这张图的人话描述；也是 ``vision.summary`` 的派生源头。
                             处理失败时写 "（本图识别失败，未产生视觉结论：<原因>）"。
        observations:        观测列表（部位 / 现象 / 极性 + 缺陷四要素）；
                             处理失败时为空列表。
        limitations:         没看清什么（"未核验"清单）；OCR 没生效、处理失败的原因也写在这。
    """

    natural_description: str = ""
    observations: list[Observation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


__all__ = ["Observation", "Polarity", "Severity", "VisionImage"]
