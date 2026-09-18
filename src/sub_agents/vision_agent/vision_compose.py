"""结论组装层：把逐图结论拼成"给人读的文本"和"给机器读的盒子"（**纯函数层**）。

职责：
    · ``compose_description()``：多图结论 → 一段 ``visual_description`` 文本；
    · ``ref_label()``：图片引用 → 图名锚点（``leak_close.png``）；
    · ``failed_image()``：把一张失败的图变成带原因的占位结论；
    · ``FAILED_PREFIX``：判定"这张图失败了"的唯一标志。

约定：
    · 本层**不依赖 LangGraph / 大模型 / Pillow / Tesseract** —— 只吃已经算好的
      ``VisionImage``，所以想改文本格式时不必碰模型和图片 IO。
    · ``compose_description`` 是 ``visual_description`` 的**唯一写入者**：
      文本永远由结构化盒子派生，两者不会各说各话。
"""

from __future__ import annotations

from src.schemas.vision import VisionImage

#: 失败占位的描述前缀（``basis`` 字段已删，下游靠这个前缀识别"这张没识别成功"）
FAILED_PREFIX = "（本图识别失败"


def failed_image(exc: BaseException) -> VisionImage:
    """把一张**处理失败**的图变成"占位结论"，而不是让异常炸掉整批。

    参数：
        exc: 捕获到的异常对象（取类型名 + 消息作为失败原因）。

    返回：
        ``VisionImage``：``natural_description`` 以 :data:`FAILED_PREFIX` 开头、
        ``observations=[]``、``limitations=["处理失败：<异常类型>: <消息>"]``。
    """
    reason = f"{type(exc).__name__}: {exc}"
    return VisionImage(
        natural_description=f"{FAILED_PREFIX}，未产生视觉结论：{reason}）",
        observations=[],
        limitations=[f"处理失败：{reason}"],
    )


def ref_label(ref: str) -> str:
    """从图片引用里取"图名"，用作描述文本里的锚点。

    参数：
        ref: 图片路径（``/a/b/pump.png``）或 URL（``http://host/img/pump.png``）。

    返回：
        最后一段的名字（``pump.png``）；取不到时原样返回 ref。
    """
    tail = str(ref).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return tail or str(ref)


def compose_description(refs: list[str], images: list[VisionImage]) -> str:
    """串联每张图结论（写进 ``visual_description``）。

    参数：
        refs:   清洗后的图片清单（与 ``images`` **等长同序**，只用于取图名做锚点）。
        images: 各图的 ``VisionImage``。

    返回：
        一段文本；单图一段，多图用空行分隔。每段形态：

            [图1] leak_close.png
            <natural_description>
            本图未核验：<limitations 用"；"连接>

        没有描述正文的图会被跳过；没有 limitations 的图不会出现"本图未核验"行。
        全部为空时返回空串 ""。
    """
    blocks: list[str] = []
    for idx, image in enumerate(images, 1):
        body = (image.natural_description or "").strip()
        if not body:
            continue
        label = ref_label(refs[idx - 1]) if idx <= len(refs) else f"图{idx}"
        block = f"[图{idx}] {label}\n{body}"
        limits = [str(x).strip() for x in (image.limitations or []) if str(x).strip()]
        if limits:
            block += f"\n本图未核验：{'；'.join(limits)}"
        blocks.append(block)
    return "\n\n".join(blocks)
