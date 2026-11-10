"""智能体：接收原始查询，完成重写与知识检索，返回检索结果。

流程是**确定性编排**（不是 tool-calling），至多两轮检索：

    原始查询
      ├─ query_rewrite.rewrite_query      1 次 LLM 调用 → N 条子问题
      ├─ retriever.retrieve_multi          N 次向量检索（召回 + 重排）→ 去重合并
      ├─ manual_review.assess_retrieval    0 次 LLM 调用，规则初筛
      │    └─ 命中规则才 → review_retrieval  1 次 LLM 调用：够不够用？缺什么？
      │         └─ 判「不足」且给了新查询 → retrieve_multi + merge_rounds 补查一轮
      └─ retriever.format_multi_result     → 给 LLM 读的上下文文本

为什么不做成 tool-calling 智能体：每一步的先后关系是确定的（不重写就没得检索、没检索就
没结果可审阅），交给 ReAct 循环只会换来不确定的调用次数、不确定的耗时和更难复现的问题。
审阅虽然带条件分支，但**分支条件是代码判的，不是模型决定的**——模型只回答「够不够」，
不决定「要不要再查」。

本模块**只负责「按什么顺序调用谁」**：不写提示词、不碰向量库、不做格式化、不建模型实例、
不实现审阅逻辑。逻辑一旦长到超过编排，就该考虑是不是放错了模块（LLM 与提示词回
manual_llm.py，重写回 query_rewrite.py，审阅回 manual_review.py，检索回 retriever.py）。
依赖方向固定为：

    manual_agent ─┬─> query_rewrite ──> manual_llm
                  ├─> manual_review ──> retriever / manual_llm
                  ├─> retriever ──────> knowledge_base
                  └─> manual_llm

用法：

    from manual_agent import run, answer_context

    # 结构化结果：重写详情 + 片段（含分数、出处）+ 给 LLM 的文本
    result = run("HCP-80-50-200A 泵组噪声大还报警，怎么排查？平时维护注意什么？")
    print(result.summary())
    for doc in result.documents:
        print(doc.metadata["section_path"], doc.metadata["retrieval_score"])

    # 只要那段文本（直接塞进生成提示词）
    context = answer_context("轴承温度超过多少会触发报警？")

直接跑本文件可看完整链路效果（会加载 bge-m3 并读 chroma_db，首次较慢）：

    python manual_agent.py

⚠️ 本模块**不生成答案**：它返回的是检索结果（原文片段 + 出处），「基于片段作答」
属于下一层（生成模块 / 调用方），不在本模块内。
"""

import time

from typing import List, Optional

from pydantic import BaseModel, Field

from manual_llm import DEFAULT_MAX_SUB_QUESTIONS
from manual_review import (
    ReviewDecision,
    assess_retrieval,
    review_note,
    review_retrieval,
)
from query_rewrite import (
    QueryRewriteResult,
    format_sub_questions,
    rewrite_query,
)
from retriever import (
    DEFAULT_TOP_K_PER_QUERY,
    MultiRetrievalResult,
    format_multi_result,
    merge_rounds,
    retrieve_multi,
)

# 重写失败（模型没配 / 调用失败）时的提示：链路已经降级成「按原查询检索」，
# 结果仍然可用，所以只要一句能读懂的话，不要把整段异常塞进 LLM 的上下文。
MAX_ERROR_CHARS = 60

# 检索轮数上限（含首轮）。第二轮已经是**基于真实召回反馈**的调整，比第一轮有信息；
# 若还没覆盖，说明知识库里大概率真的没有那部分内容——再拆只是换个说法查同一片语义空间，
# 收益递减而成本（1 次 LLM + 1 批重排）线性增长。此时正确的动作是如实告诉用户缺什么
# （`ReviewDecision.missing` 正好能带出去），而不是继续烧调用。
MAX_RETRIEVAL_ROUNDS = 2


class AgentResult(BaseModel):
    """一次智能体调用的完整结果。

    `context` 是给 LLM 读的文本，`rewrite` / `retrieval` 是给程序看的结构化数据——
    两者都在，因为下游既要「把文本塞进提示词」，也可能要「拿分数和出处做展示 / 评估」。
    只留文本会逼调用方去解析字符串，只留结构会逼调用方自己拼文本。

    `error` 非空表示这一步没走通（查询为空 / 检索失败）；`rewrite.degraded` 单独看——
    重写降级不算整体失败，检索照常进行，只是召回面变窄。

    `review` 为 `None` 表示**规则判定充足、根本没送审**（期望的多数情况）；非 `None` 时
    用 `review.reviewed` 区分「模型给了结论」与「审阅不可用降级」。`retrieval_rounds`
    是实际跑了几轮（1 或 2），可直接用来观察二次检索的触发率。
    """

    query: str = ""
    rewrite: QueryRewriteResult = Field(default_factory=QueryRewriteResult)
    retrieval: MultiRetrievalResult = Field(default_factory=MultiRetrievalResult)
    review: Optional[ReviewDecision] = None
    retrieval_rounds: int = 1
    context: str = ""
    error: str = ""
    elapsed_ms: float = 0.0

    @property
    def sub_questions(self) -> List[str]:
        return self.rewrite.questions

    @property
    def documents(self) -> list:
        return self.retrieval.documents

    @property
    def refused_queries(self) -> List[str]:
        """知识库里没有相关内容的那些子问题——可直接用来告诉用户「哪部分答不了」。"""
        return list(self.retrieval.refused_queries)

    @property
    def degraded(self) -> bool:
        """重写是否降级（降级后仍是一次正常检索，只是 query 没被拆开）。"""
        return self.rewrite.degraded

    @property
    def reviewed(self) -> bool:
        """这次有没有拿到模型的有效审阅结论（规则判定充足 / 没送审都为 False）。"""
        return self.review is not None and self.review.reviewed

    def summary(self) -> str:
        """一行摘要，便于打日志。"""
        parts = [
            f"原查询「{self.query[:20]}」",
            f"重写 {self.rewrite.summary()}",
            f"检索 {self.retrieval.summary()}（{self.retrieval_rounds} 轮）",
        ]
        if self.review is not None:
            parts.append(f"审阅 {self.review.summary()}")
        parts.append(f"总耗时 {self.elapsed_ms:.0f}ms")
        if self.error:
            parts.append(f"⚠️ {self.error}")
        return " | ".join(parts)


def _brief(error: str, limit: int = MAX_ERROR_CHARS) -> str:
    """截短失败原因：完整异常已经由 `[WARN]` 打到日志里，喂给 LLM 的上下文不该再塞一份。"""
    error = (error or "").strip()
    return error if len(error) <= limit else error[:limit] + "…"


def run(
    query: str,
    history: Optional[str] = None,
    max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
    top_k: int = DEFAULT_TOP_K_PER_QUERY,
    use_review: bool = True,
) -> AgentResult:
    """智能体主入口：原始查询 → 重写 → 检索 → 审阅 →（必要时）二次检索 → 返回。

    参数：
    - `history`：上一轮对话原文 / 摘要（可选），透传给重写器消解指代（「它的阈值是多少」）；
    - `max_sub_questions`：最多拆几条子问题（也决定了首轮的检索次数）；
    - `top_k`：每条 query 各取几条片段；
    - `use_review`：是否启用审阅与二次检索。默认开；置 False 可省掉审阅那次 LLM 调用，
      用于「只想要一次检索」的调用方或做对照实验。

    **不抛异常**：查询为空直接返回空结果；重写失败由 `rewrite_query` 自行降级成单条查询；
    检索失败转成 `"检索失败：..."` 文本放进 `context`；审阅失败降级为「审阅不可用」
    并保留现有结果。理由同前——查询失败该如实告知用户，而不是让整条链路崩掉。

    返回的 `context` 无论成功失败都非空，可直接交给生成模块使用。
    """
    started = time.perf_counter()
    query = (query or "").strip()
    if not query:
        return AgentResult(
            query="",
            context="检索失败：查询为空，请提供要检索的问题。",
            error="查询为空",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    rewrite = rewrite_query(
        query, history=history, max_sub_questions=max_sub_questions
    )

    try:
        retrieval = retrieve_multi(rewrite.questions, k=top_k)
    except Exception as exc:  # noqa: BLE001 —— 检索失败也要有可读的返回，不能抛
        error = f"{type(exc).__name__}: {exc}"
        print(f"[WARN] 知识库检索失败（{error}）")
        return AgentResult(
            query=query,
            rewrite=rewrite,
            context=f"检索失败：{_brief(error)}。请如实告知用户本次未能检索到资料。",
            error=error,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    review: Optional[ReviewDecision] = None
    rounds = 1

    if use_review:
        # ① 规则初筛：0 次 LLM 调用。返回空 = 判定充足，**连审阅都不必发起**
        hits = assess_retrieval(retrieval)
        if hits:
            # ② 可疑才请模型看：够不够？缺什么？该换什么角度补查？
            review = review_retrieval(query, rewrite.questions, retrieval, hits)

            # ③ 判「不足」且给了新查询，才跑第二轮。补查失败不影响首轮结果——
            #    `merge_rounds` 的并集性质也保证第二轮只会让候选池变大（见 retriever）。
            if review.retry_worthy and rounds < MAX_RETRIEVAL_ROUNDS:
                rounds = 2
                print(
                    f"[INFO] 审阅判定不足，用 {len(review.next_queries)} 条新查询补查一轮"
                )
                try:
                    extra = retrieve_multi(review.next_queries, k=top_k)
                    retrieval = merge_rounds(retrieval, extra)
                except Exception as exc:  # noqa: BLE001 —— 补查失败就退回首轮
                    print(
                        f"[WARN] 二次检索失败（{type(exc).__name__}: {exc}），保留首轮结果"
                    )
                    rounds = 1

    return AgentResult(
        query=query,
        rewrite=rewrite,
        retrieval=retrieval,
        review=review,
        retrieval_rounds=rounds,
        context=format_multi_result(retrieval) + review_note(review),
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def answer_context(
    query: str,
    history: Optional[str] = None,
    max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
    top_k: int = DEFAULT_TOP_K_PER_QUERY,
    use_review: bool = True,
) -> str:
    """只要给 LLM 读的那段上下文文本（重写情况、片段、审阅结论都已在其中）。"""
    return run(
        query,
        history=history,
        max_sub_questions=max_sub_questions,
        top_k=top_k,
        use_review=use_review,
    ).context


# =====================================================================
# 尚未实现（后续迭代方向，写在这里以免被当成已完成）
# =====================================================================
# 1. 答案生成：本模块只到「检索结果」为止，没有把 context 交给 LLM 生成答案并标注引用；
#    （`manual_llm.get_llm()` 已经备好裸模型，缺的是提示词与这一步的编排）
# 2. 检索充足性判断：不判断召回够不够（例如全部低置信时是否该直接拒答、还是换个说法重试）；
# 3. 多轮衔接：`history` 靠调用方传，没有会话状态；
# 4. 路由：没有「简单问题不重写直接检索」的分支——目前任何查询都要花一次 LLM 调用；
# 5. 并行检索：N 条子问题是顺序检索的（单次 ~25-40ms，条数少时够用，多了该并发）；
# 6. 重排：`retriever.rerank_documents` 目前是占位实现，检索结果仍按向量相似度排序。


if __name__ == "__main__":
    demo_query = (
        "HCP-80-50-200A 泵组运行时噪声大、振动也超标，还报了警，"
        "该怎么排查？平时维护要注意什么？"
    )
    demo_result = run(demo_query)
    print(format_sub_questions(demo_result.rewrite))
    print()
    print(demo_result.summary())
    print()
    print(demo_result.context)
