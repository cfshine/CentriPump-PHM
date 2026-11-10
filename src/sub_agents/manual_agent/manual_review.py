"""检索结果审阅：判断召回够不够用，不够就指出该换个什么角度再查一次。

**在线链路的第四步**（见《在线检索链路设计.md》三、目标流程）：

    重写 → 召回 → 重排 → **审阅** →（不够则）二次检索 → 返回

本模块只做「判断 + 决定」，不越界：不检索（`retriever.py` 的事）、不拼给 LLM 的上下文
（`retriever.py` 的事）、不编排轮次（`manual_agent.py` 的事）。它拿到一个
`MultiRetrievalResult`，回答两个问题：

    1. 这些片段够不够用？        → `assess_retrieval`（规则）+ `review_retrieval`（模型）
    2. 不够的话，该补查什么？     → `ReviewDecision.next_queries`

模块分工（依赖方向：manual_agent → manual_review → retriever / manual_llm）：

| 职责 | 入口 | 位置 |
| :-- | :-- | :-- |
| 审阅提示词 / 输出契约 / 审阅链 | `REVIEW_PROMPT` / `ReviewOutput` / `get_reviewer` | manual_llm.py |
| **规则初筛 + 调模型审阅 + 结论** | `assess_retrieval` / `review_retrieval` | 本模块 |
| 二次检索与跨轮合并 | `retrieve_multi` / `merge_rounds` | retriever.py |
| 编排（决定跑几轮） | `run` | manual_agent.py |

**设计原则：规则先行，可疑才叫模型**（见设计文档 5.6）。规则是零成本的，模型调用不是——
所以先用 `RetrievalReport`/`MultiRetrievalResult` 里的现成指标筛一遍，把「其实够用」的
查询挡在 LLM 之外。实测多数提问在第一轮就能判定充足。

用法：

    from manual_review import assess_retrieval, review_retrieval

    hits = assess_retrieval(result)          # 0 次 LLM 调用，返回命中的规则名
    if hits:
        decision = review_retrieval(query, sub_questions, result, hits)
        if decision.retry_worthy:
            ...                                  # 用 decision.next_queries 再检索一轮

⚠️ **审阅是增强手段，不是必经的关卡**：模型没配、调用失败、输出为空，一律降级为
「审阅不可用」，**保留现有结果**并如实说明，不拒答、不中断。这与重写的降级口径一致。
"""

import time

from typing import List, Optional

from pydantic import BaseModel, Field

from langchain_core.runnables import Runnable

from knowledge_base import RETRIEVAL_LOW_CONFIDENCE
from manual_llm import ReviewOutput, get_reviewer
from retriever import MultiRetrievalResult, format_multi_result


# =====================================================================
# 一、规则初筛（0 次 LLM 调用）
# =====================================================================
# 灰区：过了拒答闸门、但低于「确信相关」的区间。上界 0.60 有实测依据——
# 相关问题在本语料上的余弦下限是 0.609（见建库侧文档 5.7），落在灰区的说明只是勉强相关，
# 值得让模型看一眼。下界直接用拒答阈值：低于它的 query 在召回阶段就已经被拒了。
GREY_ZONE_LOW = RETRIEVAL_LOW_CONFIDENCE
GREY_ZONE_HIGH = 0.60

# 多数方向落空的比例（子问题数 >= 2 时才判）。留成常量是因为它**需要按真实数据标定**：
# 1/2 是拍的，还没测过触发率。
MAJORITY_REFUSED_RATIO = 0.5

# 审阅器最多给几条新查询。给多了等于把「换一个角度」变成「再拆一遍」，成本翻倍。
MAX_NEXT_QUERIES = 3


def _top1_cosine(result: MultiRetrievalResult) -> float:
    """合并结果里最高的**余弦**分。

    认余弦而不是重排分：闸门与「相关性高低」的绝对判断都建立在余弦上（二者量纲不同，
    见 retriever.rerank_documents）。重排分只用于排序，不能拿来和阈值比。
    """
    scores = [
        float(doc.metadata.get("retrieval_score", 0.0) or 0.0)
        for doc in result.documents
    ]
    return max(scores) if scores else 0.0


def assess_retrieval(result: MultiRetrievalResult) -> List[str]:
    """规则初筛：返回命中的规则名列表，**空列表 = 规则判定充足、不必送审**。

    四条规则各自回答一个不同的问题，命中哪条直接决定了下一步该往哪查：

    | 规则 | 条件 | 说明什么 |
    | :-- | :-- | :-- |
    | `empty` | 合并后一条片段都没有 | 一个方向都没查到 |
    | `all_refused` | 所有 query 都低置信 | 全部方向落空，多半是问的东西库里没有 |
    | `majority_refused` | 落空比例 >= 1/2 且子问题数 >= 2 | 拆解可能偏了——部分方向根本不存在 |
    | `grey_zone` | 最高余弦落在 `[0.50, 0.60)` | 过了闸门但只是勉强相关 |

    ⚠️ **刻意不含 `head_gap`**：向量召回的 `head_gap` 实测本来就很小（0.005~0.057），
    拿它当触发条件会导致几乎每次都命中，规则初筛就失去了「把多数查询挡在 LLM 之外」的
    意义。等有实测分布再说（设计文档 5.6 / 待定决策点 21）。
    """
    hits: List[str] = []

    if not result.documents:
        hits.append("empty")

    total = len(result.queries)
    refused = len(result.refused_queries)
    if total and refused >= total:
        hits.append("all_refused")
    elif total >= 2 and refused / total >= MAJORITY_REFUSED_RATIO:
        hits.append("majority_refused")

    if result.documents and GREY_ZONE_LOW <= _top1_cosine(result) < GREY_ZONE_HIGH:
        hits.append("grey_zone")

    return hits


# =====================================================================
# 二、审阅结论
# =====================================================================
class ReviewDecision(BaseModel):
    """一次审阅的完整结论——规则与模型两个来源合并进同一个结构。

    调用方因此不必关心「这次是谁判的」，只需要看 `retry_worthy`。

    ⚠️ **`sufficient` 的语义要说清楚**：它只在 `reviewed` 为真时有意义。
    规则判定充足、或审阅不可用时，都保持 `True`——因为**没拿到「不够」的结论就不该
    拦截结果**。审阅是增强手段而非关卡；真要看「这次有没有被审过」，看 `reviewed`。
    """

    rule_hits: List[str] = Field(default_factory=list)
    llm_consulted: bool = False
    sufficient: bool = True
    reason: str = ""
    missing: List[str] = Field(default_factory=list)
    next_queries: List[str] = Field(default_factory=list)
    error: str = ""
    elapsed_ms: float = 0.0

    @property
    def reviewed(self) -> bool:
        """这次到底有没有拿到模型的有效结论（没送审 / 送审失败都为 False）。"""
        return self.llm_consulted and not self.error

    @property
    def retry_worthy(self) -> bool:
        """是否应当触发二次检索。

        三个条件缺一不可：① 模型确实给出了结论（不是降级产物）；② 它判了「不够」；
        ③ 它给了可执行的新查询。少任何一条，再查一轮都无处下手。
        """
        return self.reviewed and not self.sufficient and bool(self.next_queries)

    def summary(self) -> str:
        """一行摘要，便于打日志。

        失败原因截断到 60 字——完整异常在 `error` 里，摘要只用来「扫一眼有没有出问题」。
        """
        if not self.llm_consulted:
            head = f"规则判定充足（{'、'.join(self.rule_hits)}）"
        elif self.error:
            brief = self.error if len(self.error) <= 60 else self.error[:60] + "…"
            head = f"审阅不可用 | {brief}"
        else:
            verdict = "充足" if self.sufficient else f"不足（补查 {len(self.next_queries)} 条）"
            head = f"审阅：{verdict}"
        return f"{head} | {self.elapsed_ms:.0f}ms"


# =====================================================================
# 三、调模型审阅
# =====================================================================
def _clean_queries(queries: List[str], exclude: List[str]) -> List[str]:
    """清洗模型给的新查询：去空、去重、去掉与已有子问题重复的、截到上限。

    与 `query_rewrite._clean_sub_questions` 同款兜底——**提示词里写了规则不代表模型会守**，
    不能只在 prompt 里约束。`exclude` 是已有的子问题：原样重说一遍的查询不产生新召回，
    只会白白多花一次检索。
    """
    def norm(text: str) -> str:
        return "".join((text or "").split()).rstrip("？?。.！!；;，,")

    seen = {norm(q) for q in exclude}
    cleaned: List[str] = []
    for query in queries or []:
        query = (query or "").strip()
        key = norm(query)
        if not query or key in seen:
            continue
        seen.add(key)
        cleaned.append(query)
        if len(cleaned) >= MAX_NEXT_QUERIES:
            break
    return cleaned


def review_retrieval(
    query: str,
    sub_questions: List[str],
    result: MultiRetrievalResult,
    rule_hits: List[str],
    llm: Optional[Runnable] = None,
) -> ReviewDecision:
    """请模型看一眼检索结果，判断够不够用；不够就给出补查的查询。

    参数：
    - `query`：用户原话。模型要拿它当「该回答什么」的基准；
    - `sub_questions`：这一轮实际用过的检索语句。给模型是为了让它**别重复问同样的角度**；
    - `result`：待审阅的检索结果，内部用 `format_multi_result` 渲染成文本喂给模型；
    - `rule_hits`：规则命中的原因，透传进结论里便于回看「模型是在什么提示下判的」；
    - `llm`：注入自定义链（测试用；传了就不走 `get_reviewer()`）。

    **不抛异常**：模型未配置、调用超时、输出为空、schema 校验失败，一律降级成
    「审阅不可用」（`error` 非空、`llm_consulted=False`、`sufficient=True`），
    由调用方保留现有结果并如实说明。

    ⚠️ **不做有界重试**，与重写不同：重写是每问必经的一环，失败则整条链路退化成
    「拿原查询硬查」；审阅只在可疑路径上跑，失败的结果只是「少了层确认」，
    而调用的代价却是实打实的一轮。收益对不上，所以不做。
    """
    started = time.perf_counter()
    rule_hits = list(rule_hits or [])

    try:
        chain = llm if llm is not None else get_reviewer()
        output = chain.invoke(
            {
                "query": query,
                "sub_questions": "\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, 1)),
                "retrieval": format_multi_result(result),
            }
        )
        if output is None:
            raise ValueError("模型未调用结构化输出工具（返回空）")
        if not isinstance(output, ReviewOutput):  # 注入自定义链时的宽松兼容
            output = ReviewOutput.model_validate(output)
        next_queries = _clean_queries(output.next_queries, sub_questions)
    except Exception as exc:  # noqa: BLE001 —— 降级，不能打断整条链路
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(f"[WARN] 检索结果审阅失败（{type(exc).__name__}: {exc}），本次保留现有结果")
        return ReviewDecision(
            rule_hits=rule_hits,
            llm_consulted=False,
            sufficient=True,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_ms=elapsed_ms,
        )

    return ReviewDecision(
        rule_hits=rule_hits,
        llm_consulted=True,
        sufficient=bool(output.sufficient),
        reason=(output.reason or "").strip(),
        missing=[m.strip() for m in (output.missing or []) if m and m.strip()],
        next_queries=next_queries,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def review_note(decision: Optional[ReviewDecision]) -> str:
    """给 LLM 读的「本次审阅结论」补充说明；不需要说明时返回空串。

    为什么要把结论回灌给生成模型：它拿到的是片段，看不到「这些片段够不够」这层判断。
    不告诉它，它就会把勉强相关的片段当成完整资料照常作答——正是 5.7 要防的事。

    三种情况分开说，因为对生成模型的要求完全不同：
    - 规则判定充足 → **什么都不说**（没审过就是没审过，不该假装审过）；
    - 审阅判定不足 → 明确列出缺什么，要求如实告知用户哪部分答不了；
    - 审阅不可用 → 说明「未经审阅」，提示可能不完整，但不要据此拒答。
    """
    if decision is None or not decision.llm_consulted:
        return ""

    if decision.error:
        return (
            "\n\n⚠️ 说明：本次检索结果**未经审阅**（审阅环节不可用），"
            "覆盖可能不完整。若片段不足以回答，请如实告知用户，不要凭已有知识补充。"
        )

    if decision.sufficient:
        return ""

    missing = "；".join(decision.missing) if decision.missing else "部分信息"
    return (
        f"\n\n⚠️ 审阅结论：上述片段**不足以完整回答**该问题，缺少：{missing}。"
        f"已尝试补查但仍未覆盖。请**如实告知用户哪部分知识库中没有**，"
        f"不要把现有片段当成完整资料作答，更不要凭已有知识补齐。"
    )
