"""单图管道：一个图片引用 → 一个 ``VisionImage`` 结论（**不读写 State**）。

职责（``analyze_one`` 的处理顺序，★ 顺序本身就是设计）：
    ① 读字节 → ② 判文件头（坏格式在花钱之前拦下）→ ③ 在**原图**上跑 OCR →
    ④ 体检 OCR 文本（噪声/有用）→ ⑤ 缩到 API 体积限制 →
    ⑥ 调视觉模型（每张图**必须**过模型，OCR 只是辅助）→ ⑦ 如实记录"OCR 未生效"

分层位置（依赖单向）：
    vision_nodes.py     编排层：批量、失败隔离、写回 State
    vision_pipeline.py  ← 本文件：管道层：单图从 ref 到结论
    vision_client.py    模型层：提示词 + 视觉调用 + 重试
    image_io.py / ocr_runner.py / ocr_quality.py  工具层：字节与文字

约定：
    · 本层是**纯函数**：不读也不写 State，返回值只有那张图的结论。
    · 原始图片字节与 Base64 绝不外传，只在函数内部活着。
"""

from __future__ import annotations

from src.schemas.state import DiagnosisState
from src.sub_agents.vision_agent.vision_models import VisionImage
from src.sub_agents.vision_agent.image_io import (
    detect_content_type,
    read_image_bytes,
    resize_for_api,
)
from src.sub_agents.vision_agent.ocr_quality import assess_ocr
from src.sub_agents.vision_agent.ocr_runner import run_ocr
from src.sub_agents.vision_agent.vision_client import understand_image
from src.sub_agents.vision_agent.vision_compose import ref_label


def analyze_one(ref: str, state: DiagnosisState) -> VisionImage:
    """单张图的完整处理管道（纯函数、不写 State）。

    参数：
        ref:   图片路径或 http(s) URL。
        state: 主图状态（只读 device_id / alarm_code 用于提示词）。

    返回：
        ``VisionImage`` —— 这张图的结论（自然描述 + 观测 + 未核验项）。

    处理顺序（★ 顺序本身就是设计）：
        ① 读字节 —— 本地路径或 http(s) 都在 ``read_image_bytes`` 里判
        ② **判文件头**（魔数）确认是 JPG/PNG/GIF/WebP；不是就直接抛
           ``ValueError: 不支持的图片格式（需要 …）`` —— **在 OCR 之前**，
           所以坏格式不会白跑一遍 Tesseract（实测 0.1 ms 就被拦下）
        ③ 在**原图**上跑 OCR（字越小越怕缩放，先从最高清像素里认字）
        ④ 体检 OCR 文本：只回答一个问题 —— **噪声还是有用的信息**
        ⑤ 分两条出口：
             · credible（有用）→ OCR 文本塞进提示词，图 + 文字一起给模型
             · 噪声/没有     → 不把垃圾喂给模型，纯看图
        ⑥ 缩放去够 API 体积限制（已在限内则**原样直传**，不重新编码）
        ⑦ 调视觉大模型（每张图**必须**过模型；OCR 只是辅助）
        ⑧ OCR 没跑成时，把"OCR 未生效 + 原因"如实写进 limitations

    失败：
        读图 / 格式不支持在**花钱之前**抛出；模型失败会重试（见
        ``vision_client.understand_image``），重试用尽才抛出。
        异常最终由 ``vision_node`` 兜成占位结论，不影响其他图。
    """
    original = read_image_bytes(ref)
    content_type = detect_content_type(original)     

    ocr = run_ocr(original)
    ocr_text = ocr.text
    assess = assess_ocr(original, ocr_text)

    if not ocr.ok:
        
        print(f"[Vision] {ref_label(ref)} OCR 未生效（{ocr.reason}）→ 直接走视觉模型")

    # 只有"有用信息"才配当模型的参考；"ww Wink" 这种噪声不进提示词
    prompt_ocr = ocr_text if assess.credible else ""
    if not assess.credible and ocr_text:
        print(f"[Vision] {ref_label(ref)} OCR 文本判为噪声（{assess.chars} 字），丢弃不喂模型")

    data = resize_for_api(original)
    content_type = detect_content_type(data)  
    image = understand_image(data, content_type, prompt_ocr, state)

    limitations = list(image.limitations)
    if not ocr.ok:
        limitations.append(
            f"OCR 未生效（{ocr.reason}）：本图未经文字提取，结论全部来自视觉模型"
        )
    return VisionImage(
        natural_description=image.natural_description,
        observations=list(image.observations),
        limitations=limitations,
    )
