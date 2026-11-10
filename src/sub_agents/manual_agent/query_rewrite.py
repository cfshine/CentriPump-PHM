"""查询重写：把用户的原始查询拆解重写成多条**自洽、详细、可直接检索**的子问题。

**本模块只做重写**——不检索、不合并、不生成答案，输入的原始查询换成一组子问题就结束。
拿到子问题之后干什么（逐条检索、合并、生成）由编排层决定，见 manual_agent.py。

为什么需要重写：用户提问是「对话式」的——一句话里塞了好几个意图、大量指代
（「它」「这个报警」「坏了」），而向量检索是「单意图、靠字面语义」的——一条 query
只能往一个语义方向去找，长句会被平均成一个模糊的向量，命中率反而下降。所以先拆解，
再让每条子问题各自去检索。

模块分工（依赖方向单向：manual_agent → query_rewrite / retriever → knowledge_base，
且 query_rewrite → manual_llm）：

| 职责 | 入口 | 位置 |
| :-- | :-- | :-- |
| LLM 客户端 / 提示词 / 输出契约 | `get_llm` / `get_rewriter` / `SubQuestion` | manual_llm.py |
| 建库（离线 / 运维，重且写库） | `KnowledgeBase.build` | knowledge_base.py |
| 检索（query → 片段，含多查询合并） | `search_knowledge_base` / `search_knowledge_base_multi` | retriever.py |
| **重写（原始查询 → 子问题）** | `rewrite_query` / `rewrite_to_list` | 本模块 |
| 编排（串起上面几步） | `run` | manual_agent.py |

本模块**不含任何 LLM 概念**：模型配置、提示词、输出契约都在 manual_llm.py，这里只有
「拿到模型输出之后怎么校验、清洗、降级」。所以换模型、改提示词都不该动这个文件——
反过来，如果一个改动需要在这里写 prompt 或者模型参数，说明放错层了。

用法：

    from query_rewrite import rewrite_query, rewrite_to_list, format_sub_questions

    # 结构化：还带每条子问题的意图与关键词，便于排查重写质量
    result = rewrite_query("HCP-80-50-200A 泵组报警了，噪声大还振动，怎么排查？平时维护注意什么？")
    for sub in result.sub_questions:
        print(sub.question, sub.intent, sub.keywords)

    # 只要一组 query 字符串
    queries = rewrite_to_list("报警码有哪些？它们各自的阈值是多少？")

    # 给人看
    print(format_sub_questions(result))

直接跑本文件可看重写效果（不加载 embedding、不碰向量库；但会调一次 LLM）：

    python query_rewrite.py

⚠️ 本模块**不做**的事（留给后续迭代，见文末「尚未实现」）：
多轮对话历史管理、子问题覆盖度自检、重写结果缓存。
"""

import time

from typing import List, Optional

from pydantic import BaseModel, Field

from langchain_core.runnables import Runnable

# `SubQuestion` / `RewriteOutput` 定义在 manual_llm.py（它们是提示词的机器可读形式，
# 必须和 SYSTEM_PROMPT 一起改），这里 import 进来有两个作用：一是本模块自己要用，
# 二是让既有调用方 `from query_rewrite import SubQuestion` 继续可用（re-export）。
from manual_llm import (
    DEFAULT_MAX_SUB_QUESTIONS,
    DEFAULT_REWRITE_ATTEMPTS,
    HARD_MAX_SUB_QUESTIONS,
    RewriteOutput,
    SubQuestion,
    get_rewriter,
)


# =====================================================================
# 一、结果模型
# =====================================================================
class QueryRewriteResult(BaseModel):
    """一次查询重写的结果。

    `degraded=True` 表示重写**没成功**（模型未配置 / 调用失败 / 输出为空），此时
    `sub_questions` 里是「原查询本身」这一条，链路仍可继续——检索质量下降，但不会
    因为重写失败而整个会话中断，且 `error` 里写明原因，绝不静默。
    """

    original_query: str = ""
    sub_questions: List[SubQuestion] = Field(default_factory=list)
    degraded: bool = False
    error: str = ""
    elapsed_ms: float = 0.0

    @property
    def questions(self) -> List[str]:
        """只要子问题文本，方便直接喂给检索。"""
        return [sub.question for sub in self.sub_questions]

    def summary(self) -> str:
        """一行摘要，便于打日志。

        失败原因截断到 60 字：摘要的用途是「扫一眼知道有没有出问题」，完整异常在
        `error` 字段里，也不该让一行日志变成几百字。
        """
        head = "降级（按原查询检索）" if self.degraded else f"{len(self.sub_questions)} 条子问题"
        error = (self.error or "").strip()
        brief = error if len(error) <= 60 else error[:60] + "…"
        return f"{head} | {self.elapsed_ms:.0f}ms" + (f" | {brief}" if brief else "")


# =====================================================================
# 二、重写入口
# =====================================================================
def _normalize(text: str) -> str:
    """去重用的归一化：忽略空白与结尾标点，「轴承温度?」与「轴承温度」视为同一条。"""
    return "".join((text or "").split()).rstrip("？?。.！!；;，,")


def _clean_sub_questions(
    subs: List[SubQuestion],
    max_sub_questions: int,
    original_query: str,
) -> List[SubQuestion]:
    """代码侧兜底：去掉空条与重复条、截到上限。

    提示词里写了规则不代表模型会守，所以这里再滤一遍——不能只在 prompt 里约束。
    """
    cleaned: List[SubQuestion] = []
    seen = {_normalize(original_query)}
    for sub in subs or []:
        question = (sub.question or "").strip()
        if not question:
            continue
        key = _normalize(question)
        if key in seen:
            continue
        seen.add(key)
        sub.question = question
        sub.keywords = [k.strip() for k in (sub.keywords or []) if k and k.strip()]
        cleaned.append(sub)
        if len(cleaned) >= max_sub_questions:
            break
    return cleaned


def _degraded(original_query: str, error: str, elapsed_ms: float) -> QueryRewriteResult:
    """重写失败时的降级结果：退回原查询，并把原因如实带出去。"""
    return QueryRewriteResult(
        original_query=original_query,
        sub_questions=[
            SubQuestion(question=original_query, intent="重写失败，按原查询直接检索", keywords=[])
        ],
        degraded=True,
        error=error,
        elapsed_ms=elapsed_ms,
    )


def rewrite_query(
    user_input: str,
    history: Optional[str] = None,
    max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
    llm: Optional[Runnable] = None,
    max_attempts: int = DEFAULT_REWRITE_ATTEMPTS,
) -> QueryRewriteResult:
    """把原始查询拆解重写成多条子问题，返回结构化结果。

    参数：
    - `history`：上一轮对话的原文 / 摘要（可选）。**追问场景必须给**，否则「它的阈值是多少」
      这类问题里的「它」无从消解——重写器只看得到当前这一句；
    - `max_sub_questions`：子问题条数上限，钳制在 `[1, HARD_MAX_SUB_QUESTIONS]`。
      ⚠️ 注意提示词里的条数上限是 `manual_llm.REWRITE_PROMPT` 用默认值固定好的，
      传别的值只影响这里的代码侧截断，见 manual_llm.py 文末的说明；
    - `llm`：注入自定义链（测试用；传了就不走 `get_rewriter()`）；
    - `max_attempts`：尝试次数上限（含首次），见下「有界重试」。

    **有界重试**：只针对「模型返回 `None`」这一种失败——它没发起结构化输出调用、
    直接回了空。这是**安全网**而非修复手段：重写失败率的高峰（约 25%）另有根因，
    是 `RewriteOutput` 类名曾带前导下划线被端点转义（见 `manual_llm.RewriteOutput`
    的类文档），改名后实测 12/12 成功。
    **其他失败不重试**：端点 400/401、超时、schema 校验不过、子问题被清洗空——
    这些重试也没用，只会把一次失败拖成三次，照旧直接降级。
    ⚠️ `OutputParserException` 也归在「不重试」里：它通常由**配置**（如工具名不合法）
    引起而非采样，重试治不了，只会掩盖问题。

    **不抛异常**：空输入、模型未配置、调用超时、结构化输出校验失败、连续重试仍失败，
    一律降级成「原查询一条」并打 `[WARN]`——重写是提升召回的手段，不是必经的关卡，
    它失败不该让整条问答链路中断；但也绝不假装成功（`degraded` / `error` 会如实带出，
    且 `error` 里写明用了几次尝试）。

    结构化结果是给程序看的；要展示给人看用 `format_sub_questions`。
    """
    started = time.perf_counter()
    query = (user_input or "").strip()
    if not query:
        return _degraded("", "查询为空", (time.perf_counter() - started) * 1000)

    max_sub_questions = max(1, min(int(max_sub_questions), HARD_MAX_SUB_QUESTIONS))
    max_attempts = max(1, int(max_attempts))
    history_text = (
        f"（上一轮对话，仅用于消解指代，不要直接改写它）\n{history.strip()}\n\n"
        if history and history.strip()
        else ""
    )

    try:
        chain = llm if llm is not None else get_rewriter()

        output = None
        for attempt in range(1, max_attempts + 1):
            output = chain.invoke({"query": query, "history": history_text})
            if output is not None:
                break
            # 重试要留痕：这不是静默的内部细节，它是「模型这一轮不听话」的证据，
            # 也是日后判断该不该调整 max_attempts 的唯一依据。
            if attempt < max_attempts:
                print(
                    f"[WARN] 重写第 {attempt}/{max_attempts} 次尝试模型返回空"
                    f"（未调用结构化输出工具），重试"
                )

        if output is None:
            raise ValueError(
                f"模型连续 {max_attempts} 次未调用结构化输出工具（均返回空）"
            )

        if not isinstance(output, RewriteOutput):  # 注入自定义链时的宽松兼容
            output = RewriteOutput.model_validate(output)
        subs = _clean_sub_questions(output.sub_questions, max_sub_questions, query)
        if not subs:
            raise ValueError("模型没有产出有效的子问题")
    except Exception as exc:  # noqa: BLE001 —— 失败要降级，不能打断整条链路
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(f"[WARN] 查询重写失败（{type(exc).__name__}: {exc}），本次按原查询直接检索")
        return _degraded(query, f"{type(exc).__name__}: {exc}", elapsed_ms)

    return QueryRewriteResult(
        original_query=query,
        sub_questions=subs,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def rewrite_to_list(
    user_input: str,
    history: Optional[str] = None,
    max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
) -> List[str]:
    """只要子问题文本的便捷入口（重写失败时返回 `[原查询]`，长度恒 ≥ 1）。"""
    return rewrite_query(
        user_input, history=history, max_sub_questions=max_sub_questions
    ).questions


def format_sub_questions(result: QueryRewriteResult) -> str:
    """把重写结果格式化成给人看的文本（调试 / 日志 / 展示用）。"""
    lines = [f"原查询：{result.original_query}"]
    if result.degraded:
        lines.append(f"⚠️ 重写降级：{result.error}")
    for index, sub in enumerate(result.sub_questions, 1):
        lines.append(f"[{index}] {sub.question}")
        if sub.intent:
            lines.append(f"    意图：{sub.intent}")
        if sub.keywords:
            lines.append(f"    关键词：{'、'.join(sub.keywords)}")
    lines.append(f"（{result.summary()}）")
    return "\n".join(lines)


# =====================================================================
# 尚未实现（后续迭代方向，写在这里以免被当成已完成）
# =====================================================================
# 1. 多轮对话历史管理：`history` 靠调用方传，本模块不维护会话状态；
# 2. 子问题覆盖度自检：不检查「拆出来的几条是否覆盖了原问题的全部意图」；
# 3. 重写结果缓存：同一句话重复问会重复调模型。


if __name__ == "__main__":
    demo_query = (
        "HCP-80-50-200A 泵组运行时噪声大、振动也超标，还报了警，"
        "该怎么排查？平时维护要注意什么？"
    )
    print(format_sub_questions(rewrite_query(demo_query)))
