"""OCR 引擎调用：图 → 文字 + "是否跑成"的状态（本模块**不依赖 LangGraph、不依赖大模型**）。

定位：**OCR 只做辅助**。本模块只管"把字认出来"；
     "认出来的字是噪声还是有用信息"由同目录 ``ocr_quality.py`` 判断。

依赖：
    · ``pytesseract``：**可选依赖**，没装也能 import 本模块 ——
      问题会在 ``run_ocr()`` 里被兜成一个 ``ok=False`` 的结果，而不是抛异常。
    · ``Pillow``：在 ``run_ocr()`` 内部延迟导入，理由同上。

约定：
    · ``run_ocr`` **永不抛异常**：OCR 是可选增强，它挂掉不该影响主流程。
"""

from __future__ import annotations

import io
from dataclasses import dataclass


@dataclass(frozen=True)
class OcrResult:
    """一次 OCR 调用的**结果 + 状态**（区别于"成功但图片本来没字"）。

    字段：
        text:   认出来的文本；失败时为空串。
        ok:     True  = Tesseract 正常跑完（text 可能为空，表示图上确实没字）。
                False = **OCR 没跑成**（没装 pytesseract、tesseract 二进制/字库缺失、
                        图片解码失败…）。此时 text 为空，原因在 reason。
        reason: ok=False 时的原因（含异常类型与消息）；ok=True 时为空串。

    为什么要把"没字"和"没跑成"分开：
        两者都是空串，但含义完全相反 ——
        前者说明"这是张照片，没有文字可读"，后者说明"**OCR 没生效**，
        本次没有任何文字信息"。后者必须如实向上反馈，否则会静默降级。
    """

    text: str = ""
    ok: bool = True
    reason: str = ""


def run_ocr(data: bytes) -> OcrResult:
    """调用 Tesseract（传统 OCR 引擎）把图片里的文字认出来。

    参数：
        data: 图片字节（**建议传原图**：字越小，缩放越伤识别率）。

    返回：
        ``OcrResult``：
            · 跑成功  → ``ok=True``，``text`` 是认出来的纯文本（已 strip）；
                        **text 可能是空串**，那表示"这张图本来就没字"（照片），不是失败。
            · 没跑成  → ``ok=False``，``text=""``，``reason`` 写明原因
                        （没装 pytesseract / tesseract 二进制或字库缺失 / 图片解码失败…）。
        **本函数永不抛异常**：OCR 是可选增强，它挂掉不该影响主流程。
    """
    try:
        import pytesseract
    except ImportError:
        return OcrResult(ok=False, reason="未安装 pytesseract（pip install pytesseract）")

    errors: list[str] = []
    for lang in ("chi_sim+eng", None):
        try:
            from PIL import Image  # 延迟导入：没装 Pillow 时也要能 import 本模块
            img = Image.open(io.BytesIO(data))
            if lang is None:
                text = pytesseract.image_to_string(img)
            else:
                text = pytesseract.image_to_string(img, lang=lang)
        except Exception as exc:  # noqa: BLE001 缺字库、坏图都算这次尝试失败
            errors.append(f"{lang or '默认语言'}: {type(exc).__name__}: {exc}")
            continue
        return OcrResult(text=(text or "").strip(), ok=True)

    return OcrResult(ok=False, reason="；".join(errors))
