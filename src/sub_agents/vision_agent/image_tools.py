"""图像 I/O 与 OCR 文本体检（本模块**不依赖 LangGraph、不依赖大模型**）。

职责：读路径或 URL → 按魔数判 MIME → 缩放去够 API 限制 → 拼 data URL
      → 跑 OCR（Tesseract）→ 体检 OCR 文本可信度 → 用 OCR 校正读数。

约定：
    · 原始图片字节与 Base64 **绝不写进 State**，只作为函数返回值在进程内传递。
    · 本模块的函数都是"纯工具"，节点层（vision_nodes.py）负责编排与回写状态。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from src.schemas.vision import Observation


def _pil_image():
    """延迟取 Pillow 的 ``Image`` 类（本模块唯一的第三方图像依赖）。

    参数：
        无。

    返回：
        ``PIL.Image`` 模块对象。
    """
    from PIL import Image
    return Image


# =============================================================================
# OCR 文本体检
# -----------------------------------------------------------------------------
#
# 现在的定位：**OCR 只做辅助**。
#   体检只回答一个问题：这段文字是**噪声**还是**有用信息**？
#     credible = 有数字 + 有单位/位号/字段名 → 是"有用信息"：
#                ① 塞进提示词给大模型当参考（增强）
#                ② 用它的数字校正模型给的读数
# =============================================================================

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

# 请求体留余量：DeepSeek 限制 48 MiB，这里把单图压到数 MB
DEFAULT_MAX_SIDE = 2048
DEFAULT_MAX_BYTES = 8_000_000


def read_image_bytes(ref: str) -> bytes:
    """按"本地路径或 http(s) URL"读出图片的原始字节。

    参数：
        ref: 图片来源。http/https 开头走网络下载，其余当本地文件路径。

    返回：
        图片文件的原始字节（未做任何解码、缩放、格式转换）。

    失败（都会抛异常，不吞）：
        ValueError:        ref 为空或全是空白。
        FileNotFoundError: 本地路径不存在（**注意：这个异常会掀翻整张 LangGraph**）。
        urllib.error.URLError / OSError: 网络不可达或超时（30 秒）。
    """
    raw = (ref or "").strip()
    if not raw:
        raise ValueError("image ref 为空")

    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        with urlopen(raw, timeout=30) as resp:  # noqa: S310 现场/测试可控 URL
            return resp.read()

    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"找不到图片文件: {raw}")
    return path.read_bytes()


def detect_content_type(data: bytes) -> str:
    """按文件头（魔数）判断图片真实 MIME 类型，**不信任扩展名**。

    参数：
        data: 图片原始字节。

    返回：
        "image/jpeg" / "image/png" / "image/gif" / "image/webp" 之一。

    失败：
        ValueError: 空数据，或文件头不属于上面四种格式
                    （例如把 HTML、txt 或损坏文件当图片传进来）。
    """
    if not data:
        raise ValueError("空图像数据")
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("不支持的图片格式（需要 JPEG / PNG / GIF / WebP）")


def _open_rgb(data: bytes):
    """把任意格式/通道的图片解成 **RGB** 模式，交给 Pillow 继续处理。

    参数：
        data: 图片原始字节。

    返回：
        ``PIL.Image.Image``（mode 恒为 "RGB"）。动图只取第一帧。

    为什么需要它：
        PNG 可能是 RGBA/调色板带透明通道，直接转 JPEG 会报错；
        这里统一铺白底（透明像素变成白色），保证后面一定存得下去。

    失败：
        PIL.UnidentifiedImageError / OSError —— 字节不是可解码图片时抛出。
    """
    Image = _pil_image()
    img = Image.open(io.BytesIO(data))
    if getattr(img, "is_animated", False):
        img.seek(0)
    if img.mode == "RGB":
        return img
    if img.mode in {"RGBA", "LA"} or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def resize_for_api(
    data: bytes,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> bytes:
    """把图片缩到"够发给大模型"的体积与边长，避免撑爆请求体。

    参数：
        data:      图片原始字节。
        max_side:  最长边像素上限，超过就等比缩小（默认 2048）。
        max_bytes: 输出体积上限，单位字节（默认 8 MB；DeepSeek 请求体上限 48 MiB，
                   这里留足余量是为了控制延迟和费用）。

    返回：
        够发给大模型的图片字节。
        **已在限制内时返回原字节（逐字节相同）**，格式自然与输入一致；
        只有超限才重新编码，那时**格式可能与输入不同**：
        JPEG/GIF/WebP 输出 JPEG；PNG 先试 PNG，若压不到上限就转 JPEG 并逐步降质量。
        ⚠ 因此调用方拿到返回值后要**重新** ``detect_content_type``（两种情况都安全）。

    快速通道（★ 2026-09-17 加，实测 9 张样例图 9/9 命中）：
        只用 ``Image.open()`` 读**文件头**拿尺寸（不解码像素，微秒级），
        若「最长边 ≤ max_side」且「体积 ≤ max_bytes」且格式是 JPEG/PNG
        → **原样返回，一次都不重新编码**。

        为什么必须加：PIL 的 ``save(format="PNG", optimize=True)`` 极慢
        （要尝试多组压缩参数），实测对**本来就不需要缩放**的 1~3MB PNG
        要花 **0.7~4.6 秒**，占整张图处理时间的 **32%** —— 纯白工。
        原样返回还顺带保住了原图的 alpha 通道（``_open_rgb`` 会把透明压成白底）。

        GIF/WebP 不走快速通道：它们原本会被转成 JPEG（顺带取第一帧），
        保持老路径以免行为变化。

    失败：
        Pillow 缺失 → ModuleNotFoundError；字节不是图片 → PIL/OSError。
    """
    Image = _pil_image()
    content_type = detect_content_type(data)

    # —— 快速通道：只读文件头，不解码像素 ——
    with Image.open(io.BytesIO(data)) as probe:
        width, height = probe.size
    if (
        len(data) <= max_bytes
        and max(width, height) <= max_side
        and content_type in {"image/jpeg", "image/png"}
    ):
        return data

    img = _open_rgb(data)
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    fmt = "JPEG" if content_type in {"image/jpeg", "image/gif", "image/webp"} else "PNG"
    quality = 85
    while True:
        buf = io.BytesIO()
        if fmt == "JPEG":
            img.save(buf, format="JPEG", quality=quality, optimize=True)
        else:
            img.save(buf, format="PNG", optimize=True)
        out = buf.getvalue()
        if len(out) <= max_bytes or (fmt == "JPEG" and quality <= 40):
            return out
        if fmt == "PNG":
            fmt = "JPEG"
            quality = 80
            continue
        quality -= 10


def to_data_url(data: bytes, content_type: str | None = None) -> str:
    """把图片字节编成 ``data:<mime>;base64,<...>``，供大模型接口内联读取。

    参数：
        data:         图片字节（**建议传缩放后的**，否则 base64 后体积涨约 1/3）。
        content_type: 已判好的 MIME；传 None 时本函数自己按魔数判。

    返回：
        data URL 字符串，形如 ``data:image/png;base64,iVBORw0KG...``。
    """
    import base64

    mime = content_type or detect_content_type(data)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def evaluate_ocr_quality(
    *,
    chars: int,
    digits: int,
    units: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    keys: tuple[str, ...] = (),
) -> OcrAssessment:
    """**纯函数**：把量好的文本特征判成"有用信息 / 噪声"，不碰图片、不碰 Tesseract。

    参数（全部关键字传参，都是"已经量好的数字"，方便单测直接构造）：
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
                       → True  = **有用信息**：塞进提示词给模型当参考 + 用它校正读数
                       → False = **噪声**（真实照片常吐 "ww Wink" 这类）：直接丢弃

    三条判据为什么是这三条：
        · 字符数：挡掉零散的 OCR 噪声；
        · 数字：纯字母噪声（"Wink"）不算信息，我们要的是读数与位号；
        · 单位/位号/字段名：证明这段文字**属于工业语境**，而不是随机纹理。

    ★ 这里**不再有** "text_dominant / 跳过视觉大模型" 的判定（2026-09-17 用户拍板）：
      OCR 只做辅助。每条图都会经过视觉大模型，OCR 有用时帮上忙、没用时被丢弃。
    ★ 行数 / Tesseract 置信度也**已从体检里删除**（2026-09-17）：它们曾靠
      ``image_to_data`` 再跑一遍 Tesseract 才能拿到，却**不参与任何判断**，
      实测白花 0~1.7 秒/图（占 12%）。真要这两个数，跑批脚本可以按需自己量。
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

    ★ 本函数**不再跑第二遍 Tesseract**（原 ``_text_layout`` 已删）：
      以前它为了"行数 + 平均置信度"两个诊断数字而重新调一次
      ``pytesseract.image_to_data``，实测 0~1.7 秒/图，且那两个数不参与判断。
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


# =============================================================================
# OCR 读数校正
# -----------------------------------------------------------------------------
# 设计意图不变：**数字以 OCR 为准**（大模型"看"数字不如传统 OCR 认字准）。
# =============================================================================

_UNIT_ALIASES = {
    "mpa": "mpa", "kpa": "kpa", "bar": "bar", "pa": "pa",
    "m3/h": "m3/h", "m³/h": "m3/h", "m3h": "m3/h", "m³h": "m3/h",
    "r/min": "rpm", "rpm": "rpm", "rpmn": "rpm",
    "kw": "kw", "hz": "hz",
    "°c": "c", "℃": "c", "c": "c", "度": "c",
    "mm/s": "mm/s", "m/s": "m/s", "m/s2": "m/s2", "mm": "mm", "m": "m",
    "l/min": "l/min", "m3": "m3", "v": "v", "a": "a", "%": "%",
}
_UNIT_CHARS = r"[A-Za-zµ℃°³/%·]"
#: needle 后面紧跟的读数：可选空白/冒号/等号 → 数字 → 可选单位 token
_TAIL_READING = re.compile(
    rf"[^\S\n]*[:：=＝]?[^\S\n]*(?<![\w.-])(\d+(?:\.\d+)?)[^\S\n]*(?P<unit>{_UNIT_CHARS}{{0,8}})"
)
_HAS_DIGIT = re.compile(r"\d")


def _norm_unit(unit: str) -> str:
    """把单位写法归一（大小写、空格、全角符号差异都抹掉），供比较用。

    参数：
        unit: 单位原文，如 "MPa"、"m³/h"、"℃"、"RPM"。

    返回：
        归一化后的字符串（如 "mpa"、"m3/h"、"c"、"rpm"）；
        表里没有的写法原样返回（小写去空格版）。
    """
    key = (unit or "").strip().lower().replace(" ", "").replace("^", "")
    return _UNIT_ALIASES.get(key, key)


def _units_compatible(model_unit: str, ocr_unit: str) -> bool:
    """判断模型给的单位与 OCR 读到的单位是否"可以互相印证"。

    参数：
        model_unit: 大模型写的单位（可能为空串）。
        ocr_unit:   OCR 行里读到的单位（可能为空串）。

    返回：
        True  = 兼容（两边归一化后相等），**或者任一侧为空**
                —— 空表示"无从判断"，此时放行（不阻塞校正）。
        False = 两边都非空且不相等（如模型 MPa、OCR bar）
                → 调用方**放弃覆盖**，只记一条"未采信"，
                  避免把 4.2 bar 当成 4.2 MPa 写进去（错 10 倍）。
    """
    left, right = _norm_unit(model_unit), _norm_unit(ocr_unit)
    if not left or not right:
        return True
    return left == right


def _append_note(text: str, note: str) -> str:
    """把一条溯源说明追加到已有文本后面（空文本时直接返回说明）。

    参数：
        text: 原有文本（如观测的 evidence），可为空串。
        note: 要追加的说明，如 "OCR 校正：出口压力 0.42 MPa"。

    返回：
        拼接后的文本；原本有内容时用中文分号"；"分隔。
    """
    return f"{text}；{note}" if text else note


def apply_ocr_readings(observations: list[Observation], ocr_text: str) -> list[Observation]:
    """用 OCR 行里的读数**校正**模型给的数字（确定性代码，不调模型）。

    参数：
        observations: 大模型给出的观测列表（每个观测可带一个 quantity）。
        ocr_text:     可信的 OCR 文本（噪声文本不会传进来）；空串时原样返回。

    返回：
        **深拷贝**后的新观测列表。调用方传进来的对象不会被改动。
        每条观测三种可能结果：
            1) 校正成功：quantity.value 改成 OCR 的数字，evidence 追加 "OCR 校正：<该行原文>"
            2) 单位冲突：value 不动，evidence 追加 "OCR 读数 X 与模型单位 Y 不一致，未采信"
            3) 不动    ：找不到读数名、或名字后面不是数字 → 完全保留模型的值
    """
    if not ocr_text or not observations:
        return observations

    result: list[Observation] = []
    for obs in observations:
        fixed = obs.model_copy(deep=True)
        q = fixed.quantity
        needle = (q.name or "").strip() if q else ""
        if q is None or not needle or _HAS_DIGIT.search(needle):
            result.append(fixed)
            continue

        for line in ocr_text.splitlines():
            pos = line.find(needle)
            if pos < 0:
                continue
            match = _TAIL_READING.match(line[pos + len(needle):])
            if not match:  # 紧邻位置没有读数 → 不猜，交给模型
                break
            value, unit = float(match.group(1)), match.group("unit") or ""
            if _units_compatible(q.unit, unit):
                q.value = value
                fixed.evidence = _append_note(fixed.evidence, f"OCR 校正：{line.strip()}")
            else:
                fixed.evidence = _append_note(
                    fixed.evidence, f"OCR 读数 {value}{unit} 与模型单位 {q.unit} 不一致，未采信"
                )
            break
        result.append(fixed)
    return result


def run_ocr(data: bytes) -> OcrResult:
    """调用 Tesseract（传统 OCR 引擎）把图片里的文字认出来。

    参数：
        data: 图片字节（**建议传原图**：字越小，缩放越伤识别率）。

    返回：
        ``OcrResult``：
            · 跑成功  → ``ok=True``，``text`` 是认出来的纯文本（已 strip）；
                        **text 可能是空串**，那表示"这张图本来就没字"（照片），
                        不是失败。
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
            Image = _pil_image()
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
