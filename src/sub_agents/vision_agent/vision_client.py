"""视觉模型层：提示词 + 单张图调用 + 重试（Step 3 里**唯一**碰大模型的文件）。

职责：
    · ``load_vision_prompt()``：读 ``configs/prompts/vision_agent.yaml`` 并缓存；
    · ``understand_image()``：图（已缩放）+ 可选 OCR 文本 → ``VisionImage`` 三字段结论；
    · ``is_retryable()``：区分"重试有用"的网络抖动 与 "重试白等" 的硬错误。

为什么单独一层：
    整个 Step 3 只有这里会花钱、会受网络影响、会自带重试策略。
    把它关进一个文件之后，管道层（``vision_pipeline.py``）和编排层（``vision_nodes.py``）
    都能在不碰大模型的前提下读懂，也能被 monkeypatch 换掉做单测。
"""

from __future__ import annotations

import time
from functools import lru_cache

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from src.schemas.state import DiagnosisState
from src.schemas.vision import VisionImage
from src.sub_agents.vision_agent.image_io import to_data_url
from src.utils.config_loader import PROJECT_PATH
from src.utils.llm_client import vision_model

#: 视觉模型**调用次数**上限（含首次）。模型失败通常是网络波动或余额不足：
#: 前者重试即可，后者重试也没用（见 is_retryable，会立刻失败、不浪费时间）。
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


@lru_cache(maxsize=1)
def load_vision_prompt() -> dict:
    """读取并缓存提示词模板（system / user 两段）。

    参数：
        无（路径由模块常量 ``PROMPT_PATH`` 决定）。

    返回：
        ``{"system": <system 提示词>, "user": <user 模板>}``；
        user 模板里含 ``{device_id}`` / ``{alarm_code}`` / ``{ocr_text}`` 三个占位符，
        由 ``understand_image`` 用 ``.format()`` 填充。

    缓存：
        ``lru_cache(maxsize=1)`` —— 同进程内只读盘一次（YAML 是静态配置）。

    失败：
        文件不存在或 YAML 缺 system/user 键 → 直接抛异常（配置错要立刻暴露）。
    """
    data = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
    return {"system": data["system"], "user": data["user"]}


def is_retryable(exc: BaseException) -> bool:
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


def understand_image(
    data: bytes,
    content_type: str,
    ocr_text: str,
    state: DiagnosisState,
) -> VisionImage:
    """把一张图（可选带 OCR 文本）发给视觉大模型，拿回三字段结论（**带重试**）。

    参数：
        data:         图片字节（**缩放后**的，够 API 限制即可）。
        content_type: data 的真实 MIME（决定 data URL 前缀）。
        ocr_text:     被判为"有用"的 OCR 文本；空串表示不给模型任何文字参考。
        state:        主图状态，只读 ``device_id`` 与 ``alarm_code`` 填提示词。

    返回：
        ``VisionImage``（模型填的 natural_description / observations / limitations）。

    重试策略：
        · 最多调 ``VISION_MAX_ATTEMPTS`` 次（默认 3，含首次），退避 1s、2s；
        · **可恢复**错误 → 重试；**硬错误**（余额不足/鉴权失败/请求不合法）→ 立刻抛；
        · 结构化输出返回 None（通常是被 max_tokens 截断）→ 也重试。
        · 本函数是**视觉链路唯一的重试层**：``vision_model`` 已显式
          ``max_retries=0``（关掉底层 SDK 自己的重试），所以最坏情况可预测：
          3 次真实 HTTP 请求 + 1s/2s 退避。

    失败（重试用尽后抛出）：
        RuntimeError（结构化输出始终解析失败）或网络/鉴权异常原样抛出。
        ⚠ 绝不降级成"仅 OCR"：OCR 只做辅助，拿它冒充视觉结论比明确失败更危险。
    """
    prompt = load_vision_prompt()
    user = prompt["user"].format(
        device_id=state["device_id"] or "",
        alarm_code=state.get("alarm_code") or "NONE",
        ocr_text=ocr_text or "（无 OCR 结果）",
    )
    data_url = to_data_url(data, content_type)
    chain = vision_model.with_structured_output(VisionImage)
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
        except Exception as exc:  # noqa: BLE001 交给 is_retryable 分类
            if not is_retryable(exc) or attempt == VISION_MAX_ATTEMPTS:
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
            print("[Vision] 视觉输出解析失败 → 准备重试")

        delay = VISION_RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
        time.sleep(delay)

    raise RuntimeError(f"视觉调用失败（已重试 {VISION_MAX_ATTEMPTS} 次）：{last_error}")
