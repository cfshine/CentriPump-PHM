"""Step 3 视觉产出的强类型契约（公共契约的一部分）。

★ 2026-09-17 精简为**只有 3 个字段**（用户拍板：除了这三个，其他都删）：
    natural_description  这张图的人话描述
    observations[]       target（部位）/ finding（看见了什么）/ polarity（极性）
    limitations[]        没看清什么

被删掉的字段（曾经存在过，现在不要了）：
    source_ref     哪张图 → 改用**位置对齐**：findings.images[i] ↔ image_refs[i]
    image_kind     图片种类枚举（8 类）→ 下游暂时不需要
    basis          结论依据（vision / vision+ocr / failed）→ 失败看描述前缀
    confidence     置信度 → 没有消费者
    quantity       读数（name/value/unit）→ 连带删除了 OCR 数字校正
    evidence       依据原文 → 归入 natural_description / limitations 的文字里

为什么单独一个文件、而不是写在 ``state.py`` 里：
    ``state.py`` 反过来 import 节点模块会形成循环（``vision_nodes → state``），
    所以契约模型放在 ``schemas/`` 下，两边都能安全 import。

★ 一条硬规则（配套测试守护）：
    ``visual_description`` 只能由 ``visual_findings`` 派生，唯一写入者是主图的
    ``vision_node`` —— 同一份内容不出现"两个写入者"，就不会漂移。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: 单个观测的极性（封闭词表）：
#: abnormal=可见异常 / normal=可见正常 / unknown=看不出或无法判断
Polarity = Literal["abnormal", "normal", "unknown"]


class Observation(BaseModel):
    """一条"看见了什么"。只描述现象，不写故障原因

    字段：
        target:   画面中的对象，如 "驱动端轴承箱" / "出口压力表" / "顶部告警条"。
        finding:  看见了什么，只描述不归因（如 "压盖下缘有深色液体连续流淌"）。
        polarity: 极性，abnormal / normal / unknown（封闭词表，供下游写 if）。

    用途：
        这是整个视觉环节里最"机器友好"的部分：下游可以遍历、按极性计数、
        按部位聚合，而不需要理解散文。
    """

    target: str = Field(..., description="画面中的对象，如驱动端轴承箱、出口压力表、铭牌")
    finding: str = Field(..., description="看见了什么，只描述不归因")
    polarity: Polarity = "unknown"


class VisionImage(BaseModel):
    """**一张图**的结论（盒子 ``images`` 列表的元素）。

    字段：
        natural_description: 这张图的人话描述；也是 ``visual_description`` 的派生源头。
                             处理失败时写 "（本图识别失败，未产生视觉结论：<原因>）"。
        observations:        观测列表（部位 / 现象 / 极性）；处理失败时为空列表。
        limitations:         没看清什么（"未核验"清单）；OCR 没生效、处理失败的原因也写在这。
    """

    natural_description: str = ""
    observations: list[Observation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class VisionFindings(BaseModel):
    """视觉结构化产出（视觉节点 ``vision_node`` 的 ``findings`` 字段）。

    字段：
        images: 每张图一份 ``VisionImage``。

    两条约定：
        ① 单图也包一层 ``images`` —— **形状恒定**，下游永远写 ``findings.images`` 遍历，
           不必先判断"这次是一张还是多张、是 dict 还是 list"；空图时 ``images == []``。
        ② ``findings.images[i]`` 与 ``image_refs[i]`` **按位置一一对应**
    说明：
        本类由我方代码构造、不进模型 schema，所以开 ``extra="forbid"``
        —— 构造时字段名写错会立刻报错，而不是被静默忽略。
    """

    model_config = ConfigDict(extra="forbid")

    images: list[VisionImage] = Field(default_factory=list)


__all__ = ["Observation", "Polarity", "VisionFindings", "VisionImage"]
