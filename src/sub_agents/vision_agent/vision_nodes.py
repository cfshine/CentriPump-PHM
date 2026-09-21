"""Step 3 的**唯一节点**：``vision_node``（主图里一行 ``add_node`` 就能挂）。

分层（读代码时的地图，依赖从上往下单向）：
    vision_nodes.py     ← 本文件：**只有编排** —— 批量、失败隔离、缺陷映射、写回契约
    vision_pipeline.py  管道层：单图 ``analyze_one``（不写 State）
    vision_client.py    模型层：提示词 + 视觉调用 + 重试（★ 唯一碰大模型的地方）
    vision_compose.py   组装层：结论 ↔ 文本（纯函数，零第三方依赖）
    vision_models.py    数据模型：Observation / VisionImage（Step 3 内部，不是公共契约）
    image_io.py / ocr_runner.py / ocr_quality.py   工具层：字节与文字（不碰模型）

数据流（一行看懂）：
    context.image_refs → 逐图 analyze_one → [描述 + 观测 + 未核验]
                       → vision.summary（人读）+ vision.findings（每个缺陷一条）+ vision.status
"""

from __future__ import annotations
from src.schemas.state import DiagnosisState, VisionFinding, VisionState
from src.sub_agents.vision_agent.vision_compose import (
    FAILED_PREFIX,
    compose_summary,
    failed_image,
    ref_label,
)
from src.sub_agents.vision_agent.vision_models import VisionImage
from src.sub_agents.vision_agent.vision_pipeline import analyze_one


def _vision_update(state: DiagnosisState, **updates) -> dict:
    """把本次要改的字段合并进 ``state.vision``，返回**只含 vision 一个键**的回写字典。

    参数：
        state:   当前全局状态（读它现有的 vision 盒子）。
        updates: 本次要覆盖的 ``VisionState`` 字段（status / findings / summary）。

    返回：
        ``{"vision": VisionState}``。
    """
    return {"vision": VisionState.model_validate({**state.vision.model_dump(), **updates})}


def _derive_status(images: list[VisionImage], findings: list[VisionFinding]) -> str:
    """派生 ``VisionState.status``（封闭四态，纯确定、可单测）。

    参数：
        images:   各图结论（空列表表示本来就没有图）。
        findings: 已映射好的缺陷列表。

    返回：
        ``NO_IMAGE``     —— 没有图片
        ``FAILED``       —— 有图，但**每一张**都处理失败
        ``DEFECT_FOUND`` —— 至少一张成功，且发现了缺陷
        ``NO_DEFECT``    —— 至少一张成功，且没发现缺陷

    """
    if not images:
        return "NO_IMAGE"
    if all((img.natural_description or "").startswith(FAILED_PREFIX) for img in images):
        return "FAILED"
    return "DEFECT_FOUND" if findings else "NO_DEFECT"


def vision_node(state: DiagnosisState) -> dict:
    """**Step 3 的唯一节点**：逐张处理图片，把结果写进 ``state.vision``。

    参数：
        state: 全局状态；读 ``state.context`` 的 ``image_refs``（``list[ImageRef]``）、
               ``device_id``、``alarm_code``。

    返回：
        ``{"vision": VisionState}``：
            ``status``:   四态之一（见 :func:`_derive_status`）
            ``findings``: 每个缺陷一条 ``VisionFinding``
                          （``image_id / defect_type / severity / location / confidence / evidence``）
            ``summary``:  人读文本 —— 每图描述 + 末尾"未核验汇总"节
                          （由 ``compose_summary`` 从结构化结论派生，是唯一写入者）

    失败处理：
        单张图失败不拖垮整批（沿用既有行为）：那一张变成带失败原因的占位结论，
        不产生 finding，但失败原因**仍然出现在 summary 里**（不静默吞掉）。
    """
    ctx = state.context
    refs = list(ctx.image_refs or [])

    if not refs:
        print("[Vision] 无 image_refs，跳过 Step 3")
        return _vision_update(state, status="NO_IMAGE", findings=[], summary="")

    print(f"[Vision] 待处理 {len(refs)} 张图")
    images: list[VisionImage] = []
    for ref in refs:
        try:
            image = analyze_one(ref.uri, state)
        except Exception as exc:  # noqa: BLE001 单张失败不许拖垮整批
            image = failed_image(exc)
            print(f"[Vision] {ref_label(ref.uri)} → 处理失败 {type(exc).__name__}: {exc}")
        else:
            abnormal = sum(1 for o in image.observations if o.polarity == "abnormal")
            print(
                f"[Vision] {ref_label(ref.uri)} → obs={len(image.observations)} "
                f"abnormal={abnormal} 限制项={len(image.limitations)}"
            )
        images.append(image)

    # —— 缺陷映射：每个 abnormal 观测 → 一条 VisionFinding（D3：不完整的跳过并留痕）——
    findings: list[VisionFinding] = []
    for ref, image in zip(refs, images):
        for obs in image.observations:
            if obs.polarity != "abnormal":
                continue
            if not (obs.defect_type.strip() and obs.severity and obs.evidence.strip()):
                image.limitations.append(
                    f"「{obs.target}」的异常缺少完整的缺陷要素"
                    f"（defect_type/severity/evidence），未计入 findings"
                )
                print(f"[Vision] {ref_label(ref.uri)} → 跳过不完整的缺陷：{obs.target}")
                continue
            findings.append(
                VisionFinding(
                    image_id=ref.image_id,
                    defect_type=obs.defect_type.strip(),
                    severity=obs.severity,
                    location=obs.target,
                    confidence=obs.confidence,
                    evidence=obs.evidence.strip(),
                )
            )

    print(f"[Vision] 缺陷 {len(findings)} 条")
    return _vision_update(
        state,
        status=_derive_status(images, findings),
        findings=findings,
        summary=compose_summary([r.uri for r in refs], images),
    )
