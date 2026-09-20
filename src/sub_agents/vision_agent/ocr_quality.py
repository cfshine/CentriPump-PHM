"""OCR 文本体检：把认出来的文字判成"噪声"还是"有用信息"。

定位：**OCR 只做辅助**。
    体检只回答一个问题：这段文字是**噪声**还是**有用信息**？
      credible = 有数字 + 有单位/位号/字段名 → 是"有用信息"：
                  塞进提示词给大模型当参考（增强）

依赖：
    · 量特征与判结论（``evaluate_ocr_quality``）是**纯函数**：零第三方依赖、
      不碰图片、不碰 Tesseract，给几个数字就出结论，单测可直接构造。
    · 只有 ``assess_ocr(text=None)`` 那一支才需要真去跑 OCR，
      它会调用同目录 ``ocr_runner.run_ocr()``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.sub_agents.vision_agent.ocr_runner import run_ocr

TEXT_MIN_CHARS = 24     # 低于此长度只当噪声（真实照片 OCR 常吐 "ww Wink" 这类）
TEXT_MIN_DIGITS = 4     # "有用信息"必须有数字（纯字母噪声不算）

_UNIT_RE = re.compile(
    r"(MPa|kPa|\bbar\b|m3/h|m³/h|r/min|rpm|kW|Hz|°C|℃|mm/s|L/min)", re.I
)
_TAG_RE = re.compile(r"\b[A-Z]{2,4}[-_ ]?\d{2,4}\b")           # FAL-104 / TAH-101 / PI-102
_KEY_RE = re.compile(
    r"\b(SP|PV|Model|Rated|Head|Speed|Power|Serial|Status|Location|Alarm|TAG"
    r"|PRESS|TEMP|VIB|Flow)\b",
    re.I,
)


@dataclass(frozen=True)
class OcrAssessment:
    """OCR 文本的体检结果（不可变，构造后只读）。

    前 5 个字段是**量出来的原始特征**（只做诊断与调阈值用，不直接参与判断）：
        chars:       去掉所有空白后的字符数（衡量文本量）。
        digits:      数字字符个数（读数图的关键特征）。
        units:       命中的单位词集合（如 mpa / m3/h / rpm）。
        tags:        命中的工位号集合（如 FAL-104 / PI-102）。
        keys:        命中的字段名集合（如 SP / PV / MODEL / ALARM）。

    最后 1 个是**结论**：
        credible:    这段文字是"有用信息"（True）还是"噪声"（False）。
                     True  → 塞进提示词给模型当参考，并用它校正读数
                     False → 直接丢弃，不喂给模型
    """

    chars: int = 0
    digits: int = 0
    units: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()
    credible: bool = False       # True=有用信息（当提示词/校正读数）；False=噪声（丢弃）


def evaluate_ocr_quality(
    *,
    chars: int,
    digits: int,
    units: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    keys: tuple[str, ...] = (),
    ) -> OcrAssessment:
    """**纯函数**：把量好的文本特征判成"有用信息 / 噪声"

    参数：
        chars:      去空白后的字符数。
        digits:     数字字符个数。
        units:      命中的单位词（如 ("mpa", "rpm")）。
        tags:       命中的工位号（如 ("FAL-104",)）。
        keys:       命中的字段名（如 ("SP", "PV", "ALARM")）。

    返回：
        ``OcrAssessment``。唯一的结论字段是 ``credible``：

            credible = 字符数 ≥ TEXT_MIN_CHARS(24)
                       且 数字个数 ≥ TEXT_MIN_DIGITS(4)
                       且 至少有 单位/位号/字段名 之一
                       → True  = **有用信息**：塞进提示词给模型当参考
                       → False = **噪声**（真实照片常吐 "ww Wink" 这类）：直接丢弃

    """
    credible = (
        chars >= TEXT_MIN_CHARS
        and digits >= TEXT_MIN_DIGITS
        and bool(units or tags or keys)
    )
    return OcrAssessment(
        chars=chars,
        digits=digits,
        units=tuple(units),
        tags=tuple(tags),
        keys=tuple(keys),
        credible=credible,
    )


def assess_ocr(data: bytes, text: str | None = None) -> OcrAssessment:
    """量特征 + 判有用/噪声：把一张图（或一段已知 OCR 文本）走完整套体检。

    参数：
        data: 图片字节。**只有 text=None 时才会用到**（那时本函数自己调
              ``run_ocr(data)``）；调用方已经 OCR 过、把文本传进来时，本函数
              **完全不碰图片、也不再跑 Tesseract**。
        text: 已经 OCR 出来的文本；传 None 表示"你来 OCR"。
              ⚠ 传空串表示"确定没有文字"；**OCR 没跑成**的情形请用
              ``run_ocr()`` 拿到 ``OcrResult.ok=False`` 自行上报（见 OcrResult 说明）。

    返回：
        ``OcrAssessment``（字符数 / 数字个数 / 单位 / 位号 / 字段名 + credible）。
        文本为空时直接返回全 0 的结果，**不调用 Tesseract**。

    失败：
        不抛异常。OCR 不可用时文本为空，自然按"噪声"处理。
    """
    raw = run_ocr(data).text if text is None else (text or "")
    if not raw.strip():
        return evaluate_ocr_quality(chars=0, digits=0)

    return evaluate_ocr_quality(
        chars=len(re.sub(r"\s+", "", raw)),
        digits=len(re.findall(r"\d", raw)),
        units=tuple(sorted({m.lower() for m in _UNIT_RE.findall(raw)})),
        tags=tuple(sorted({t.upper() for t in _TAG_RE.findall(raw)})),
        keys=tuple(sorted({k.upper() for k in _KEY_RE.findall(raw)})),
    )
