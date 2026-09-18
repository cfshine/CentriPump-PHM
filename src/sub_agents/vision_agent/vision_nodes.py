"""Step 3 的**唯一节点**：``vision_node``（主图里一行 ``add_node`` 就能挂）。

分层（读代码时的地图，依赖从上往下单向）：
    vision_nodes.py     ← 本文件：**只有编排** —— 批量、失败隔离、写回公共契约
    vision_pipeline.py  管道层：单图 ``analyze_one``（不写 State）
    vision_client.py    模型层：提示词 + 视觉调用 + 重试（★ 唯一碰大模型的地方）
    vision_compose.py   组装层：盒子 ↔ 文本（纯函数，零第三方依赖）
    image_io.py / ocr_runner.py / ocr_quality.py   工具层：字节与文字（不碰模型）

数据流（一行看懂）：
    image_refs → 逐图 analyze_one → [自然描述 + 观测 + 未核验] → 拼成一段文本 + 一个盒子
"""

from __future__ import annotations

from src.schemas.state import DiagnosisState
from src.schemas.vision import VisionFindings, VisionImage
from src.sub_agents.vision_agent.vision_compose import (
    compose_description,
    failed_image,
    ref_label,
)
from src.sub_agents.vision_agent.vision_pipeline import analyze_one

def vision_node(state: DiagnosisState):
    """**Step 3 的唯一节点**：逐张处理图片，把结果写回公共契约。

    参数：
        state: 主图状态；读 ``image_refs``（list[str]）、``device_id``、``alarm_code``。

    返回：
        ``image_refs``:         清洗后的清单（去空格、丢空项）—— 回写它是为了让
                                ``visual_findings.images[i] ↔ image_refs[i]`` **等长同序**，
                                下游能靠位置知道"第 i 条结论来自哪张图"（``source_ref`` 已删）。
        ``visual_description``: 由 ``compose_description`` 从盒子派生的整段文本（给人读）。
        ``visual_findings``:    ``VisionFindings(images=[...])``（给机器读），
                                条数与图片数一致；失败的图是带失败原因的占位结论。
    """
    refs = [str(r).strip() for r in (state.get("image_refs") or []) if str(r).strip()]
    if not refs:
        print("[Vision] 无 image_refs，跳过 Step 3")
        return {"image_refs": [], "visual_description": "", "visual_findings": VisionFindings()}

    print(f"[Vision] 待处理 {len(refs)} 张图")
    images: list[VisionImage] = []
    for ref in refs:
        try:
            image = analyze_one(ref, state)
        except Exception as exc:  # noqa: BLE001 单张失败不许拖垮整批
            image = failed_image(exc)
            print(f"[Vision] {ref_label(ref)} → 处理失败 {type(exc).__name__}: {exc}")
        else:
            abnormal = sum(1 for o in image.observations if o.polarity == "abnormal")
            print(
                f"[Vision] {ref_label(ref)} → obs={len(image.observations)} "
                f"abnormal={abnormal} 限制项={len(image.limitations)}"
            )
        images.append(image)

    return {
        "image_refs": refs,
        "visual_description": compose_description(refs, images),
        "visual_findings": VisionFindings(images=images),
    }
