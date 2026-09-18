"""节点适配层：从 State 解包 → 调用 image_tools / 视觉大模型 → 回写公共契约。

分工（读代码时的地图）：
    image_tools.py   纯工具：读图、判 MIME、缩放、OCR、体检、读数校正（不依赖大模型）
    schemas/vision.py 强类型契约：VisionImage / VisionFindings / Observation / Quantity
    本文件           唯一的"生产"环节：编排上面两者，把结果写回 State

图拓扑只有两个节点：``ingest``（清洗 + 空图短路） → ``extract``（干活）。
真正的算法都在 ``analyze_one`` 这个普通函数里，图只负责编排。
"""

from __future__ import annotations

import time
from functools import lru_cache

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.schemas.vision import (
    Basis,
    ImageKind,
    Observation,
    VisionFindings,
    VisionImage,
)
from src.sub_agents.vision_agent.image_tools import (
    OcrAssessment,
    OcrResult,
    apply_ocr_readings,
    assess_ocr,
    detect_content_type,
    read_image_bytes,
    resize_for_api,
    run_ocr,
    to_data_url,
)
from src.sub_agents.vision_agent.vision_state import VisionAgentState
from src.utils.config_loader import PROJECT_PATH
from src.utils.llm_client import vision_model

#: 视觉模型**调用次数**上限（含首次）。模型失败通常是网络波动或余额不足：
#: 前者重试即可，后者重试也没用（见 _is_retryable，会立刻失败、不浪费时间）。
VISION_MAX_ATTEMPTS = 3
#: 重试退避基数（秒）：第 1 次失败等 1s、第 2 次等 2s，避免把抖动放大成雪崩。
VISION_RETRY_BASE_DELAY_S = 1.0
#: 命中即**立刻抛出、不重试**的错误特征（重试纯属白等）。
#: 刻意列得很短：宁可多等几秒重试，也不要漏掉一次"真能成功"的调用。
_FATAL_ERROR_HINTS = (
    "insufficient balance",   # 余额不足
    "invalid api key",        # 鉴权失败
    "authentication",         # 同上（异常类名里常见）
    "permission",             # 无权限
    "model not found",        # 模型名写错
    "invalid_request_error",  # 请求本身不合法（如 schema 不被接受）
    "402",                    # 余额不足的 HTTP 码
    "401",                    # 未鉴权的 HTTP 码
)

#: 提示词模板路径（只讲业务规则，不含 JSON 格式——格式由结构化输出 schema 注入）
PROMPT_PATH = PROJECT_PATH / "configs" / "prompts" / "vision_agent.yaml"


class VisionLLMOutput(BaseModel):
    """**只给大模型填**的那部分结论（节点的结构化输出契约）。

    参数（字段含义）：
        image_kind:          图片种类，只能取 ``schemas.vision.ImageKind`` 里的枚举值。
        natural_description: 一段现场可见现象的自然语言描述（给人看的正文）。
        observations:        观测列表，每条含 target / finding / polarity / quantity / evidence。
        limitations:         看不清、被遮挡、无法判断的内容（防止过度解读）。
        confidence:          模型自评置信度，0.0~1.0。
        rag_queries:         0~5 条手册检索短词（只写可见线索，不写故障名猜测）。
    """

    image_kind: ImageKind
    natural_description: str
    observations: list[Observation]
    limitations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rag_queries: list[str] = Field(default_factory=list, max_length=5)


@lru_cache(maxsize=1)
def _load_vision_prompt() -> dict:
    """读取并缓存提示词模板（system / user 两段）。

    参数：
        无（路径由模块常量 ``PROMPT_PATH`` 决定）。

    返回：
        ``{"system": <system 提示词>, "user": <user 模板>}``；
        user 模板里含 ``{device_id}`` / ``{alarm_code}`` / ``{ocr_text}`` 三个占位符，
        由 ``_vision_understand`` 用 ``.format()`` 填充。

    """
    data = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
    return {"system": data["system"], "user": data["user"]}


def _is_retryable(exc: BaseException) -> bool:
    """判断这个异常**重试是否有意义**。

    参数：
        exc: 捕获到的异常。

    返回：
        False = 硬错误（余额不足、鉴权失败、模型名写错、请求不合法…）——
                重试只是白等，应当立刻抛出让人看到真正原因。
        True  = 其余情况（网络抖动、连接重置、超时、限流、5xx…）——
                退避后重试，多数情况第二次就成功。
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return not any(hint in text for hint in _FATAL_ERROR_HINTS)


def _vision_understand(
    data: bytes,
    content_type: str,
    ocr_text: str,
    state: VisionAgentState) -> VisionLLMOutput:
    """把一张图（可选带 OCR 文本）发给视觉大模型，拿回结构化结论。

    参数：
        data:         图片字节（**缩放后**的，够 API 限制即可）。
        content_type: data 的真实 MIME（决定 data URL 前缀；缩放后要重新判）。
        ocr_text:     可信的 OCR 文本；空串表示不给模型任何文字参考。
        state:        子图状态，只读其中 ``device_id`` 与 ``alarm_code`` 用于提示词。

    返回：
        ``VisionLLMOutput``（模型填好的枚举/描述/观测/限制/置信度/检索词）。
    """
    prompt = _load_vision_prompt()
    user = prompt["user"].format(
        device_id=state["device_id"] or "",
        alarm_code=state.get("alarm_code") or "NONE",
        ocr_text=ocr_text or "（无 OCR 结果）",
    )
    data_url = to_data_url(data, content_type)
    chain = vision_model.with_structured_output(VisionLLMOutput)
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=[
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]),
    ]

    last_error = ""
    for attempt in range(1, VISION_MAX_ATTEMPTS + 1):
        try:
            result = chain.invoke(messages)
        except Exception as exc:  # noqa: BLE001 交给 _is_retryable 分类
            if not _is_retryable(exc) or attempt == VISION_MAX_ATTEMPTS:
                raise
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[Vision] 视觉调用第 {attempt} 次失败（{last_error}）→ 准备重试")
        else:
            if result is not None:
                return result
            last_error = "结构化输出解析失败（返回 None）"
            if attempt == VISION_MAX_ATTEMPTS:
                raise RuntimeError(
                    "视觉结构化输出解析失败（返回 None）；通常是输出被 max_tokens 截断"
                )
            print(f"[Vision] 视觉输出解析失败 → 准备重试")

        delay = VISION_RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
        time.sleep(delay)

    raise RuntimeError(f"视觉调用失败（已重试 {VISION_MAX_ATTEMPTS} 次）：{last_error}")


def analyze_one(ref: str, state: VisionAgentState) -> VisionImage:
    """单张图的完整处理管道（本模块的核心函数，纯函数、不碰 State 的写操作）。

    参数：
        ref:   图片路径或 http(s) URL。
        state: 子图状态（只读 device_id / alarm_code 用于提示词）。

    返回：
        ``VisionImage`` —— 这张图的结构化结论（含描述、观测、限制、basis、confidence）。
    """
    original = read_image_bytes(ref)
    content_type = detect_content_type(original)

    ocr = run_ocr(original)
    ocr_text = ocr.text
    assess = assess_ocr(original, ocr_text)

    if not ocr.ok:
        # 如实上报：OCR 没生效 ≠ 图上没字，必须让下游看得见
        print(f"[Vision] {_ref_label(ref)} OCR 未生效（{ocr.reason}）→ 直接走视觉模型")

    # 只有"有用信息"才配当模型的参考；"ww Wink" 这种噪声不进提示词
    prompt_ocr = ocr_text if assess.credible else ""
    if not assess.credible and ocr_text:
        print(f"[Vision] {_ref_label(ref)} OCR 文本判为噪声（{assess.chars} 字），丢弃不喂模型")

    data = resize_for_api(original)
    content_type = detect_content_type(data)  # 缩放可能把 PNG 转成 JPEG，以实际字节为准
    llm = _vision_understand(data, content_type, prompt_ocr, state)
    observations = apply_ocr_readings(list(llm.observations), prompt_ocr)
    basis: Basis = "vision+ocr" if prompt_ocr else "vision"
    limitations = list(llm.limitations)
    if not ocr.ok:
        limitations.append(
            f"OCR 未生效（{ocr.reason}）：本图未经文字提取，结论全部来自视觉模型"
        )
    return VisionImage(
        source_ref=ref,
        image_kind=llm.image_kind,
        basis=basis,
        natural_description=llm.natural_description,
        confidence=llm.confidence,
        limitations=limitations,
        observations=observations,
    )


def _ref_label(ref: str) -> str:
    """从图片引用里取"图名"，用作描述文本里的锚点。

    参数：
        ref: 图片路径（``/a/b/pump.png``）或 URL（``http://host/img/pump.png``）。

    返回：
        最后一段的名字（``pump.png``）；取不到时原样返回 ref。
        Windows 风格反斜杠、结尾多余的 ``/`` 都会被处理掉。
    """
    tail = str(ref).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return tail or str(ref)


def _compose_description(images: list[VisionImage]) -> str:
    """把逐图结论拼成自然语言描述（写进 ``visual_description``）。

    参数：
        images: 各图的 ``VisionImage`` 列表（顺序即展示顺序）。

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
        block = f"[图{idx}] {_ref_label(image.source_ref)}\n{body}"
        limits = [str(x).strip() for x in (image.limitations or []) if str(x).strip()]
        if limits:
            block += f"\n本图未核验：{'；'.join(limits)}"
        blocks.append(block)
    return "\n\n".join(blocks)


def ingest_node(state: VisionAgentState):
    """子图第一个节点：清洗图片清单；**没图就短路**。

    参数：
        state: 子图状态；读 ``image_refs``（list[str]，元素前后空格会被去掉，
               空字符串项被丢弃）。

    返回（写给 State 的增量 dict）：
        没图时 → ``{"image_refs": [], "visual_description": "",
                    "visual_findings": VisionFindings()}``
                 —— 形状恒定（盒子在、列表空、描述空），下游不用判 None。
        有图时 → ``{"image_refs": <清洗后的清单>}``
                 —— 顺带把清洗结果回写，后续节点直接用。
    """
    refs = [str(r).strip() for r in (state.get("image_refs") or []) if str(r).strip()]
    if not refs:
        print("[Vision] 无 image_refs，跳过 Step 3")
        return {"image_refs": [], "visual_description": "", "visual_findings": VisionFindings()}
    print(f"[Vision] 待处理 {len(refs)} 张图")
    return {"image_refs": refs}


def _failed_image(ref: str, exc: BaseException) -> VisionImage:
    """把一张**处理失败**的图变成"占位结论"，而不是让异常炸掉整批。

    参数：
        ref: 出问题的图片引用（路径或 URL），原样写进 source_ref 便于定位。
        exc: 捕获到的异常对象（取类型名 + 消息作为失败原因）。

    返回：
        ``VisionImage``，特征如下：
            basis="failed"        —— 下游代码一句 ``basis == "failed"`` 就能挑出来
            image_kind="other"
            natural_description="（本图识别失败，未产生视觉结论：<异常类型>: <消息>）"
            confidence=0.0
            limitations=["处理失败：<异常类型>: <消息>"]
            observations=[]       —— 没有任何观测
    """
    reason = f"{type(exc).__name__}: {exc}"
    return VisionImage(
        source_ref=ref,
        image_kind="other",
        basis="failed",
        natural_description=f"（本图识别失败，未产生视觉结论：{reason}）",
        confidence=0.0,
        limitations=[f"处理失败：{reason}"],
        observations=[],
    )


def extract_node(state: VisionAgentState):
    """子图第二个节点：逐张处理，然后把**两半**结果写回公共契约。

    参数：
        state: 子图状态；读 ``image_refs``。

    返回（写给 State 的增量 dict，只有两个键）：
        ``visual_description``: 由 ``_compose_description`` 从盒子派生的整段文本（给人读）。
        ``visual_findings``:    ``VisionFindings(images=[...])``（给机器读）。
                                条数与输入图片一致；失败的图是 ``basis="failed"`` 的占位结论。
    """
    refs = [str(r).strip() for r in (state.get("image_refs") or []) if str(r).strip()]
    images: list[VisionImage] = []

    for ref in refs:
        try:
            image = analyze_one(ref, state)
        except Exception as exc:  # noqa: BLE001 单张失败不许拖垮整批（见 docstring）
            image = _failed_image(ref, exc)
            print(f"[Vision] {_ref_label(ref)} → 处理失败 {type(exc).__name__}: {exc}")
        else:
            abnormal = sum(1 for o in image.observations if o.polarity == "abnormal")
            print(
                f"[Vision] {_ref_label(ref)} → basis={image.basis} kind={image.image_kind} "
                f"conf={image.confidence} obs={len(image.observations)} abnormal={abnormal} "
                f"限制项={len(image.limitations)}"
            )
        images.append(image)

    return {
        "visual_description": _compose_description(images),
        "visual_findings": VisionFindings(images=images),
    }