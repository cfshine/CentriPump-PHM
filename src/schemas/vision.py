"""Step 3 视觉产出的强类型契约（公共契约的一部分，下游 Step4/5/7 可读）。

为什么单独一个文件、而不是写在 ``state.py`` 或 ``vision_nodes.py`` 里：
    - 写在 ``vision_nodes.py``：``state.py`` 就得反过来 import 节点模块，
      而 ``vision_nodes → vision_state → state`` 已经是一条链，会**循环导入**。
    - 写在 ``state.py``：模型定义和"全局状态契约"混在一处，职责不清。
    README 对 ``src/schemas/`` 的定义是「Pydantic 强类型数据契约：DiagnosisState
    与各 Agent 出入参模型」，所以放这里最合适，两边都能安全 import。

分层（谁读什么）：
    ``DiagnosisState.visual_description``  —— 人读：给 Step4 做 RAG 检索输入、给 Step7 写报告
    ``DiagnosisState.visual_findings``     —— 机器读：给下游代码遍历/判断/统计

★ 两条硬规则（配套测试在 tests/test_vision_agent.py）：
    1. ``visual_description`` 只能由 ``visual_findings`` 派生，唯一写入者是子图的
       ``extract_node`` —— 同一份内容不出现"两个写入者"，就不会漂移。
    2. 空图时 ``images == []`` 且描述为 ``""``，形状恒定，下游不用判空。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: 图片种类枚举（唯一封闭词表，给确定性代码做过滤/分类用）：
#: nameplate=铭牌 / gauge_or_meter=表计 / hmi_or_screenshot=屏幕截图 /
#: leak_or_stain=泄漏或污渍 / mechanical_surface=机械表面 / thermal=热成像 /
#: scene=现场全景 / other=其它
ImageKind = Literal[
    "nameplate",
    "gauge_or_meter",
    "hmi_or_screenshot",
    "leak_or_stain",
    "mechanical_surface",
    "thermal",
    "scene",
    "other",
]

#: 单个观测的极性（封闭词表）：
#: abnormal=可见异常 / normal=可见正常 / unknown=看不出或无法判断
Polarity = Literal["abnormal", "normal", "unknown"]

#: 这条结论的依据：
#:   vision    = 只看了图（OCR 没有产出可用文字，或本来就没字）
#:   vision+ocr= 看图 + **可信的** OCR 文本一起给模型（OCR 当辅助增强）
#:   failed    = **这张图处理失败了**（读不到文件、格式不支持、接口重试后仍报错…）。
Basis = Literal["vision", "vision+ocr", "failed"]


class Quantity(BaseModel):
    """一个机器可读的读数（如 出口压力 0.25 MPa）。

    字段：
        name:  读数名称，如 "出口压力" / "flow_pv" / "bearing_temperature"。
        value: 数值；模型读不出来时为 None（例如"表盘读数不可辨"）。
        unit:  单位，如 "MPa" / "m³/h" / "℃" / "r/min"；不确定时为空串。

    用途：
        让下游**代码**直接取出数值做比较（例如与 Step2 的时序指标交叉验证），
        不必去解析自然语言描述里的数字。

    ★ 本类会出现在给大模型的结构化输出 schema 里，所以**不加 extra="forbid"**
      —— 模型多吐一个键时宁可忽略，也不要整条输出校验失败。

    注意：
        契约里**没有** raw（读数原始字符串）字段：原文归 ``Observation.evidence``，
        两个字段说同一件事属于冗余。
    """

    name: str = ""
    value: float | None = None
    unit: str = ""


class Observation(BaseModel):
    """一条"看见了什么"。只描述现象，不写故障原因（归因是 Step5 的事）。

    字段：
        target:   画面中的对象，如 "驱动端轴承箱" / "出口压力表" / "顶部告警条"。
        finding:  看见了什么，只描述不归因（如 "压盖下缘有深色液体连续流淌"）。
        polarity: 极性，abnormal / normal / unknown（封闭词表）。
        quantity: 该对象的读数，没有则为 None。
        evidence: 依据原文（OCR 命中的那一行，或画面线索描述），用于溯源。

    用途：
        这是整个视觉环节里最"机器友好"的部分：下游可以遍历、按极性计数、
        按部位聚合、按读数比对，而不需要理解散文。
    """

    target: str = Field(..., description="画面中的对象，如驱动端轴承箱、出口压力表、铭牌")
    finding: str = Field(..., description="看见了什么，只描述不归因")
    polarity: Polarity = "unknown"
    quantity: Quantity | None = None
    evidence: str = Field(default="", description="依据原文（OCR 命中行或画面线索）")


class VisionImage(BaseModel):
    """**一张图**的完整结论（盒子里 ``images`` 列表的元素）。

    字段：
        source_ref:           图片来源（本地路径或 URL）；溯源锚点，"这条结论出自哪张图"。
        image_kind:           图片种类（封闭词表）。
        basis:                结论依据（vision / vision+ocr / failed）。
                              ``failed`` 表示这张图没识别成功，是一条带原因的占位结论。
        natural_description:  这张图的人话描述；也是 ``visual_description`` 的派生源头。
        confidence:           置信度 0.0~1.0，来源只有两处：
                              **模型自评**（正常识别），或 **0.0**（处理失败占位）。
                              没有"写死的中低值"了 —— OCR 只做辅助。
        limitations:          没看清什么（"未核验"清单）；报告里要交代的"未覆盖范围"。
                              处理失败的图，失败原因也写在这里。
        observations:         观测列表（部位 / 现象 / 极性 / 读数 / 依据）；失败时为空列表。

    说明：
        本类由我方代码构造、**不进模型的 schema**，所以开 ``extra="forbid"``
        —— 构造时字段名写错会立刻报错，而不是被静默忽略。
    """

    model_config = ConfigDict(extra="forbid")

    source_ref: str = ""                 # 本地路径或 URL（溯源锚点）
    image_kind: ImageKind = "other"
    basis: Basis = "vision"
    natural_description: str = ""        # 这张图的人话描述（文本派生的源头）
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)   # 没看清什么
    observations: list[Observation] = Field(default_factory=list)


class VisionFindings(BaseModel):
    """视觉结构化总盒子（写进 ``DiagnosisState.visual_findings`` 的就是它）。

    字段：
        images: 每张图一份 ``VisionImage``。

    为什么单图也要包一层 ``images``：
        **形状恒定**。下游永远写 ``findings.images`` 去遍历，
        不必先判断"这次是一张还是多张、是 dict 还是 list"。
        空图时 ``images == []``，而不是 None 或缺字段。

    用途：
        给"将来的下游代码"用（README 里规划了 Step4~7，目前尚未实现）：
        例如统计异常条数、按部位聚合、把某个读数与时序指标比对。
    """

    model_config = ConfigDict(extra="forbid")

    images: list[VisionImage] = Field(default_factory=list)


__all__ = [
    "Basis",
    "ImageKind",
    "Observation",
    "Polarity",
    "Quantity",
    "VisionFindings",
    "VisionImage",
]
