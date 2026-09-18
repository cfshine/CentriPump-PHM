"""图像 I/O 与编解码（本模块**不依赖 LangGraph、不依赖大模型**）。
职责：读路径或 URL → 按魔数判 MIME → 缩放去够 API 限制 → 拼 data URL。
在 Step 3 工具层里的位置（三层，越往下越"纯"）：
    image_io.py       ← 本文件：字节进、字节出（只认图片，不认文字）
    ocr_runner.py     图 → 文字（调 Tesseract）
    ocr_quality.py    文字 → "噪声/有用"的判断（纯函数）

约定：
    · 原始图片字节与 Base64 **绝不写进 State**，只作为函数返回值在进程内传递。
    · Pillow 是本模块唯一的第三方依赖，且在 ``_pil_image()`` 里**延迟导入** ——
      只想判个 MIME 的调用方（``detect_content_type``）不必先装 Pillow。
"""

from __future__ import annotations
import io
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
# 请求体留余量：DeepSeek 限制 48 MiB，这里把单图压到数 MB
DEFAULT_MAX_SIDE = 2048
DEFAULT_MAX_BYTES = 8_000_000

def _pil_image():
    """延迟取 Pillow 的 ``Image`` 类（本模块唯一的第三方图像依赖）。

    参数：
        无。

    返回：
        ``PIL.Image`` 模块对象。
    """
    from PIL import Image
    return Image



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
        PNG 可能是 RGBA/调色板带透明通道，当压缩图片大小时需要转格式会报错；
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
