"""检索入口：把知识库检索包装成 LangChain tool，供智能体调用。

与建库的分工——**建库不在这个模块，也不该给 LLM**：

| 场景 | 入口 | 位置 |
| :-- | :-- | :-- |
| 建库（离线 / 运维，重且写库） | `KnowledgeBase.build` | knowledge_base.py |
| 检索（在线，给 LLM 读） | `KNOWLEDGE_BASE_TOOL` / `search_knowledge_base` | 本模块 |
| 多条 query 的检索（重写后的子问题） | `search_knowledge_base_multi` | 本模块 |

**接收的是「要检索什么」，不关心这些 query 从哪来**——是用户原话，还是重写模块拆出来的
子问题，本模块都一视同仁。谁决定「检索哪几条」属于编排层的职责，见 manual_agent.py。

**两阶段检索（先多召回、再精排）**——每条 query 各自走完这两步，然后才合并：

    每条 query ──> retrieve(k=recall_k)      多召回，候选池
                └─> rerank(每条保留 top_n)      cross-encoder 成对打分，收敛
                        └─> 按 chunk_id 去重合并

为什么重排要**在合并之前**、且**逐条 query 各自做**：cross-encoder 打的是 (query, chunk)
成对分，合并之后就找不到「该拿哪条 query 去打分」了；用原始查询也不行——那正是被重写
拆解掉的那条模糊查询，等于把子问题的价值又还回去。

为什么召回量要**大于**最终保留量：重排只能在召回的候选里排序，**召回不够，重排无从发挥**。
两个参数各管一头——`recall_k` 管覆盖率，`k` 管收敛后的上下文预算。

⚠️ **拒答闸门始终认余弦分**（`retrieval_score`），重排分（`rerank_score`）只用于排序与
截断。两者的量纲与分布完全不同，理由见 `rerank_documents` 的说明。

给 LangChain 智能体用：

    from retriever import KNOWLEDGE_BASE_TOOL

    agent = create_agent(llm, tools=[KNOWLEDGE_BASE_TOOL, ...])

直接当普通函数用（调试，或不经 LLM 的调用方）：

    print(search_knowledge_base("轴承温度超过多少会触发报警"))
    print(search_knowledge_base_multi(["轴承温度报警阈值", "轴承温度高的排查步骤"]))

需要分数 / 出处做程序化处理时，用结构化入口（`search_*` 返回的是格式化文本）：

    result = retrieve_multi(["...", "..."])       # -> MultiRetrievalResult
    print(result.documents[0].metadata)

本模块只读向量库，不写、不建、不清库；`search_*` 所有异常都转成文本返回，不向智能体抛错。
"""

import time

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool

from knowledge_base import KnowledgeBase, RetrievalReport

# -------------------- 检索参数 --------------------
DEFAULT_TOP_K = 5          # 与 KnowledgeBase.retrieve 的默认 k 对齐
MAX_TOP_K = 20             # 上限：防止 LLM 传 1000 条把上下文塞爆
MAX_CHUNK_CHARS = 1200     # 单条片段上限（实测最长 chunk 779 字符，正常不会触发）
MAX_TOTAL_CHARS = 4000     # 全部片段的总预算，超出部分不展示并明确说明

# -------------------- 两阶段检索参数 --------------------
# 重排开关。默认**开**：模型不可用时不会静默退化，而是打 `[WARN]` 后整批退回向量序。
USE_RERANK = True

# 重排开启时，每条 query 先召回多少条候选（第一阶段）。实测（bge-reranker-v2-m3 fp16
# + RTX 4060）重排约 10.6 ms/对，4 条子问题 × 15 = 60 对 ≈ 0.64s，可接受。
DEFAULT_RECALL_K_PER_QUERY = 15
MAX_RECALL_K = 50          # 上限：防调用方传入超大值把一次检索变成几千对打分

_builder: Optional[KnowledgeBase] = None


def get_builder() -> KnowledgeBase:
    """检索侧共用的 `KnowledgeBase` 实例（懒加载单例）。

    `KnowledgeBase()` 本身很轻——embedding 模型与向量库都在属性/方法里懒加载，
    所以 import 本模块不会加载模型、不会碰数据库。
    """
    global _builder
    if _builder is None:
        _builder = KnowledgeBase()
    return _builder


# -------------------- 重排 --------------------
def rerank_available() -> tuple:
    """重排是否可用。返回 `(可用, 不可用原因)`，**会真的把模型加载起来**。

    存在的意义是「早失败、只失败一次、失败得响亮」：
    - 早——在进入逐条 query 的循环**之前**判定，避免每条 query 各失败一次、各打一行日志；
    - 一致——重排是**整批**开或**整批**关。若允许个别 query 失败后单独退回向量序，
      合并池里就会一半片段带 `rerank_score`、一半不带，排序变成拿 sigmoid 分比余弦分，
      结果不可解释（`_order_score` 的注释同样在守这条）。

    加载本身较慢（实测约 9~16s，含预热），但只发生一次——`KnowledgeBase.reranker`
    是进程内单例。
    """
    try:
        get_builder().reranker
        return True, ""
    except Exception as exc:  # noqa: BLE001 —— 未配模型 / 缺依赖 / 加载失败都算不可用
        return False, f"{type(exc).__name__}: {exc}"


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = DEFAULT_TOP_K,
) -> List[Document]:
    """按与 query 的相关性对候选重排，返回新的 Document 列表（前 `top_n` 条）。

    薄封装 `KnowledgeBase.rerank`（cross-encoder 对 (query, chunk) 成对打分），
    本层不加逻辑，只把调用点收在本模块里，便于统一处理可用性与降级。

    ⚠️ **重排分不能用来判「知识库有没有这个内容」**——那是余弦分（`retrieval_score`）
    的职责。`sentence_transformers` 在 `num_labels=1` 时默认给 cross-encoder 的输出套
    Sigmoid，于是 `rerank_score` 落在 `[0, 1]` 内、**看起来**像概率、也像能套
    `RETRIEVAL_LOW_CONFIDENCE = 0.5`。实测过：一条真实相关查询的 55 条候选，重排分
    最高 0.938、**均值只有 0.208**，绝大多数低于 0.5——拿它当闸门会把几乎所有候选拒掉，
    表现为「明明有资料却拒答」。

    **两者各司其职**：余弦管闸门（阈值 0.5 已按本模型 + 本语料标定），重排分管排序与
    截断（只用相对序，不设绝对阈值）。

    依赖：`RERANKER_MODEL_DIR`。不可用时由调用方（`retrieve_multi` / `search_knowledge_base`）
    统一判定并退回向量序，本函数不自行降级——否则降级点会散落在多处。
    """
    return get_builder().rerank(query, documents, top_n=top_n)


def _order_score(doc: Document) -> float:
    """合并去重时的排序分：重排过就用 `rerank_score`，否则用余弦。

    **两种分不同源、不可比**，所以绝不能在同一次检索里混用。守这条的是
    `rerank_available()` 的「整批开或整批关」判定：要么所有片段都被重排过
    （全有 `rerank_score`），要么全都没有。
    """
    metadata = doc.metadata or {}
    if "rerank_score" in metadata:
        return float(metadata["rerank_score"])
    return _score_of(doc)


# -------------------- 检索入口 --------------------
def search_knowledge_base(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    use_rerank: bool = USE_RERANK,
) -> str:
    """检索知识库，返回**带出处的原文片段文本**（给 LLM 读，因此返回 `str` 而非 Document）。

    - 每条片段都带出处：文件名 / 章节路径 / 页码或 Sheet，以及向量相似度；
    - **低置信时拒答**：`top1 < 阈值（默认 0.5）` 时**不返回任何片段正文**，只回一段
      说明。这是设计文档 5.7 的闸门——「应当触发拒答 / 转人工，而不是硬塞给 LLM 生成」，
      避免模型拿不相关内容编答案。阈值是实测标定的（相关问题 0.609~0.743 / 无关 0.447~0.448），
      换 embedding 模型或换语料需要重新标定；
    - **不抛异常**：空查询、模型没配、库不存在等一律转成 `"检索失败：..."` 文本返回——
      工具抛错会打断整个智能体会话，而这些失败都是可以如实回报的。

    返回的是格式化文本，不是结构化数据。需要分数/出处做程序化处理时，直接用
    `get_builder().retrieve()` / `evaluate_retrieval()`（见 knowledge_base.py）。

    `use_rerank` 现已真正生效（`USE_RERANK` 默认 True）：开启时先多召回、再精排收敛。
    重排不可用（模型没配 / 加载失败）时**退回单阶段**并打 `[WARN]`，不静默 no-op。
    """
    query = (query or "").strip()
    if not query:
        return "检索失败：查询为空，请提供要检索的问题。"

    try:
        top_k = int(top_k if top_k is not None else DEFAULT_TOP_K)
        top_k = max(1, min(top_k, MAX_TOP_K))     # 挡住 0 / 负数 / 超大值

        builder = get_builder()

        rerank_on = False
        if use_rerank:
            rerank_on, reason = rerank_available()
            if not rerank_on:
                print(f"[WARN] 重排不可用（{reason}），本次退回向量序单阶段检索")

        # 重排关掉时召回量必须等于 top_k，不能还用大召回量——否则是拿一堆**未精排**的
        # 候选去挤 MAX_TOTAL_CHARS 预算，比不重排更糟（见 DEFAULT_RECALL_K_PER_QUERY）。
        recall_k = DEFAULT_RECALL_K_PER_QUERY if rerank_on else top_k
        recall_k = max(top_k, min(int(recall_k), MAX_RECALL_K))

        # 闸门认**召回阶段**的余弦分：低置信判定与重排无关，见 rerank_documents。
        candidates = builder.retrieve(query, k=recall_k)
        gate = builder.evaluate_retrieval(candidates, query=query)
        if gate.low_confidence:
            return _refusal_text(query, gate)

        if rerank_on:
            documents = rerank_documents(query, candidates, top_n=top_k)
            # 展示用的报告换 rerank_score 口径，好让顶部那行分数与展示的片段一致
            report = builder.evaluate_retrieval(
                documents, query=query, score_key="rerank_score"
            )
        else:
            documents = candidates[:top_k]
            report = builder.evaluate_retrieval(documents, query=query)
    except Exception as exc:                      # noqa: BLE001
        return f"检索失败：{type(exc).__name__}: {exc}"

    if not documents:
        return _empty_text(query)
    return _format_result(query, documents, report, reranked=rerank_on)


def _source_line(metadata: dict) -> str:
    """把溯源信息拼成一行：文件名 | 章节路径 | 页码 / Sheet 名。

    只取文件名（不展开绝对路径）：调用方是 LLM，每 5 条片段都要重复一遍
    `/home/bicouper/hqyj_ai/projects/AI Industry/docs/...` 太占上下文预算。
    """
    parts = [Path(metadata.get("source", "")).name or "未知来源"]
    if metadata.get("section_path"):
        parts.append(str(metadata["section_path"]))
    if metadata.get("page") is not None:
        parts.append(f"page={metadata['page']}")
    if metadata.get("sheet_name"):
        parts.append(f"sheet={metadata['sheet_name']}")
    return " | ".join(parts)


def _format_result(
    query: str,
    documents: List[Document],
    report,
    reranked: bool = False,
) -> str:
    """把检索结果格式化成给 LLM 读的文本（带出处、带上下文预算）。

    分数标签随 `reranked` 变：重排后展示的是 `rerank_score`，标成「相似度」会误导
    阅读者把它当成余弦分去和 0.5 比（两者不可比，见 `rerank_documents`）。
    """
    score_label = "重排分" if reranked else "相似度"
    lines = [
        f"知识库检索结果｜查询：{query}",
        f"命中 {report.returned} 条｜最高{score_label} {report.top1:.3f}"
        f"｜来源 {report.distinct_sources} 份资料",
    ]

    used = 0
    for index, doc in enumerate(documents, 1):
        text = doc.page_content
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + "……（本片段已截断）"
        block = (
            f"\n[{index}] {score_label} {_order_score(doc):.3f}"
            f"｜{_source_line(doc.metadata)}\n{text}"
        )
        if used + len(block) > MAX_TOTAL_CHARS:
            # 不静默丢弃：明确说明还剩几条没展示
            lines.append(f"\n（其余 {len(documents) - index + 1} 条超出上下文预算，未展示）")
            break
        used += len(block)
        lines.append(block)

    return "\n".join(lines)


def _refusal_text(query: str, report) -> str:
    """低置信拒答：只回说明，**不给片段正文**（设计文档 5.7 的闸门）。"""
    print(
        f"[WARN] 低置信拒答：查询「{query[:20]}」top1={report.top1:.3f} "
        f"< 阈值 {report.threshold}，已按约定不返回片段"
    )
    return (
        f"知识库中没有找到与问题相关的内容。\n"
        f"查询：{query}\n"
        f"检索到的最高相似度 {report.top1:.3f}，低于置信度阈值 {report.threshold}——"
        f"按约定此时不提供片段，以免基于不相关内容编造答案。\n"
        f"请如实告知用户知识库中没有相关资料，并可以建议用户换个说法或补充背景信息；"
        f"不要凭已有知识作答。"
    )


def _empty_text(query: str) -> str:
    """命中 0 条（分数缺失等边界情况）。"""
    return (
        f"知识库中没有检索到相关内容。\n"
        f"查询：{query}\n"
        f"请如实告知用户知识库中没有相关资料，不要凭已有知识编造答案。"
    )


# -------------------- 多查询检索 --------------------
# 用途：把「一组 query」（典型来源是查询重写模块拆出的子问题）各自检索一遍，再去重合并。
# 为什么需要：一条 query 只能往一个语义方向去找；拆成多条并行召回，覆盖面才够。
DEFAULT_TOP_K_PER_QUERY = 3    # 多查询时每条 query 各取几条（比单查询小，
                               # 因为总量 = top_k × query 数，会挤占 MAX_TOTAL_CHARS 预算）
MAX_QUERIES = 8                # 单次上限：挡调用方传入几十条 query 把一次检索变成几百次


class MultiRetrievalResult(BaseModel):
    """一次多查询检索的结果（已去重合并）。

    为什么必须去重：多条 query 常常命中**同一条** chunk，重复片段既白占 LLM 的上下文
    预算，又会让它误以为「多处资料都这么说」而高估该结论的可靠性。

    `hit_by` 记录每条 chunk 是被哪条 query 命中的——子问题写得好不好，靠它来回看：
    某条 query 一条都没召回、或者所有命中都集中在一条 query 上，都说明重写有问题。

    `reports` 与 `rerank_reports` 是**同一批 query 在重排前后各评一次**的结果，
    分数量纲不同（余弦 vs 过了 Sigmoid 的 cross-encoder 分），因此只能比 `head_gap`
    这类相对指标——见 `head_gap_gain`。
    """

    queries: List[str] = Field(default_factory=list)
    documents: List[Document] = Field(default_factory=list)
    reports: List[RetrievalReport] = Field(default_factory=list)
    refused_queries: List[str] = Field(default_factory=list)
    hit_by: Dict[str, str] = Field(default_factory=dict)
    reranked: bool = False
    rerank_reports: List[RetrievalReport] = Field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def source_count(self) -> int:
        """涉及几份资料（去重后）。"""
        return len({Path(str(doc.metadata.get("source", ""))).name for doc in self.documents})

    @property
    def head_gap_gain(self) -> Optional[Tuple[float, float]]:
        """重排前后 `head_gap` 的均值 `(前, 后)`；未重排或数据不足时返回 `None`。

        为什么只比 `head_gap`：余弦 ∈ [-1, 1]、重排分 ∈ [0, 1]（cross-encoder 输出
        过了 Sigmoid），两者的**绝对水平不可比**；而 `head_gap` 是同一批候选**内部**的
        相对差距，跨阶段可比。重排正确时它应变大——这是「重排到底有没有用」的量化答案
        （现状向量召回的 head_gap 很小，实测 0.005~0.046，提升空间明确）。
        """
        if not self.reranked or not self.reports or not self.rerank_reports:
            return None
        after_by_query = {report.query: report.head_gap for report in self.rerank_reports}
        pairs = [
            (report.head_gap, after_by_query[report.query])
            for report in self.reports
            if report.query in after_by_query and report.returned
        ]
        if not pairs:
            return None
        return (
            sum(before for before, _ in pairs) / len(pairs),
            sum(after for _, after in pairs) / len(pairs),
        )

    def summary(self) -> str:
        """一行摘要，便于打日志。"""
        stage = "重排" if self.reranked else "向量"
        gain = self.head_gap_gain
        tail = f" | head_gap {gain[0]:.3f}→{gain[1]:.3f}" if gain else ""
        return (
            f"{len(self.queries)} 条 query -> 去重后 {len(self.documents)} 条片段 | "
            f"来源 {self.source_count} 个 | "
            f"低置信 {len(self.refused_queries)} 条 | {stage}{tail} | {self.elapsed_ms:.0f}ms"
        )


def _dedup_key(doc: Document) -> str:
    """合并去重的 key：优先用 `chunk_id`（入库时的确定性 id），缺失时退回定位信息。"""
    metadata = doc.metadata or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return "|".join(
        str(metadata.get(field, ""))
        for field in ("source", "section_path", "page", "sheet_name")
    ) + "|" + doc.page_content[:60]


def _score_of(doc: Document) -> float:
    return float(doc.metadata.get("retrieval_score", 0.0) or 0.0)


def retrieve_multi(
    queries: List[str],
    k: int = DEFAULT_TOP_K_PER_QUERY,
    recall_k: Optional[int] = None,
    use_rerank: bool = USE_RERANK,
) -> MultiRetrievalResult:
    """一组 query 各自走完「召回 → 精排」再按 `chunk_id` 去重合并。

    - **重复 query 先去重**（同一条查两遍没有意义，只是白花时间）；
    - **上限 `MAX_QUERIES`**：超出时**只取前 N 条并打 `[WARN]`**，不静默丢弃——
      调用方需要知道「后面的 query 这次没被检索」；
    - **逐条 query 各自评估置信度**：低置信的 query 不返回其片段（拒答闸门同样生效），
      记进 `refused_queries`。比整组拒答更有用——能看出缺的是哪个方向；
    - 同一条 chunk 被多条 query 命中时保留**最高分**那次，`hit_by` 记首个命中它的 query。

    参数：
    - `k`：**最终**每条 query 保留几条（精排后的收敛量），语义与大召回量无关；
    - `recall_k`：第一阶段召回量。`None` 时按重排是否可用自动取
      （可用 → `DEFAULT_RECALL_K_PER_QUERY`；不可用 → `k`）。
      ⚠️ 重排不可用时**必须**回落到 `k`：拿大召回量去喂未精排的候选，只会用一堆
      中低分片段挤爆 `MAX_TOTAL_CHARS` 预算，比不重排更糟；
    - `use_rerank`：是否走两阶段。模型不可用时整批退回向量序并打 `[WARN]`。

    **会抛异常**（与 `retrieve` 一致）：参数非法、库不存在、模型加载失败等，交由调用方
    决定是转成文本还是中断；要「不抛错」的版本用 `search_knowledge_base_multi`。
    """
    normalized: List[str] = []
    for query in queries or []:
        query = (query or "").strip()
        if query and query not in normalized:
            normalized.append(query)

    if not normalized:
        raise ValueError("查询列表为空")
    if len(normalized) > MAX_QUERIES:
        print(
            f"[WARN] 多查询检索收到 {len(normalized)} 条 query，超出上限 {MAX_QUERIES}，"
            f"本次只检索前 {MAX_QUERIES} 条"
        )
        normalized = normalized[:MAX_QUERIES]

    k = max(1, min(int(k if k is not None else DEFAULT_TOP_K_PER_QUERY), MAX_TOP_K))

    # 计时得绕开一次性初始化：`get_builder()` 本身很轻，真正加载 embedding 模型、
    # 打开向量库发生在第一次 `get_vector_store()`（数秒级）——先把它触发了再开始计时，
    # 否则 elapsed_ms 会把预热算成检索耗时，让「这次检索慢」变成一个查不出原因的假象。
    # 重排模型同理（加载约 9~16s，且首次前向含 kernel 编译），所以这里也先触发。
    builder = get_builder()
    builder.get_vector_store()

    # 重排可用性**一次性**判定（不是每条 query 各判一次）：整批开或整批关，
    # 否则合并池里会一半带 rerank_score、一半不带，排序成了拿 sigmoid 分比余弦分。
    rerank_on = False
    if use_rerank:
        rerank_on, reason = rerank_available()
        if not rerank_on:
            print(f"[WARN] 重排不可用（{reason}），本次退回向量序单阶段检索")

    if recall_k is None:
        recall_k = DEFAULT_RECALL_K_PER_QUERY if rerank_on else k
    recall_k = max(k, min(int(recall_k), MAX_RECALL_K))

    started = time.perf_counter()
    merged: Dict[str, Document] = {}
    hit_by: Dict[str, str] = {}
    reports: List[RetrievalReport] = []
    rerank_reports: List[RetrievalReport] = []
    refused: List[str] = []

    for query in normalized:
        # ① 多召回
        candidates = builder.retrieve(query, k=recall_k)

        # ② 闸门认召回阶段的**余弦**分——低置信判定与重排无关，见 rerank_documents
        report = builder.evaluate_retrieval(candidates, query=query)
        reports.append(report)
        if report.low_confidence:
            refused.append(query)
            continue

        # ③ 精排收敛
        if rerank_on:
            documents = rerank_documents(query, candidates, top_n=k)
            rerank_reports.append(
                builder.evaluate_retrieval(documents, query=query, score_key="rerank_score")
            )
        else:
            documents = candidates[:k]

        # ④ 合并
        for doc in documents:
            key = _dedup_key(doc)
            current = merged.get(key)
            if current is None or _order_score(doc) > _order_score(current):
                merged[key] = doc
                hit_by[key] = query

    return MultiRetrievalResult(
        queries=normalized,
        documents=sorted(merged.values(), key=_order_score, reverse=True),
        reports=reports,
        refused_queries=refused,
        hit_by=hit_by,
        reranked=rerank_on,
        rerank_reports=rerank_reports,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def merge_rounds(
    first: MultiRetrievalResult,
    second: MultiRetrievalResult,
) -> MultiRetrievalResult:
    """把两轮检索的结果合成一个（**并集 + 全局重排**），用于二次检索。

    合并规则与轮内完全一致（按 `chunk_id` 去重、保留最高分、全局按分数降序），
    只是候选池换成两轮的并集。

    ⚠️ **这个性质是设计的一部分**：第一轮的片段仍留在池子里参与最终排序，所以第二轮
    **只会让候选池变大、排序更准，不会丢掉第一轮的命中**——「二次检索不会让结果变差」
    因此是结构性成立的，不需要额外加保护（见设计文档 5.8）。

    `queries` / `refused_queries` 取两轮并集（去重、保持先后），便于回看「这一问总共用过
    哪些检索语句、哪些方向落空了」。`elapsed_ms` 是两轮之和；`reports` /
    `rerank_reports` 直接拼接（`head_gap_gain` 按 query 配对，不受影响）。

    ⚠️ 跨轮比较 `rerank_score` 仍有量纲问题（分属不同 `(query, chunk)` 对）。在「只用它
    排序、不用它判有无」的前提下可接受——拒答一律由**各轮召回阶段的余弦闸门**决定。
    """
    merged: Dict[str, Document] = {}
    hit_by: Dict[str, str] = {}
    for result in (first, second):
        for doc in result.documents:
            key = _dedup_key(doc)
            current = merged.get(key)
            if current is None or _order_score(doc) > _order_score(current):
                merged[key] = doc
                hit_by[key] = result.hit_by.get(key, "")

    def _uniq(values: List[str]) -> List[str]:
        seen: List[str] = []
        for value in values:
            if value not in seen:
                seen.append(value)
        return seen

    return MultiRetrievalResult(
        queries=_uniq(list(first.queries) + list(second.queries)),
        documents=sorted(merged.values(), key=_order_score, reverse=True),
        reports=list(first.reports) + list(second.reports),
        refused_queries=_uniq(list(first.refused_queries) + list(second.refused_queries)),
        hit_by=hit_by,
        # 两轮用同一个 use_rerank，所以正常恒等；写 or 是防御——若一轮带重排分、
        # 一轮不带，_order_score 会拿 Sigmoid 分比余弦分，那种排序不可解释。
        reranked=first.reranked and second.reranked,
        rerank_reports=list(first.rerank_reports) + list(second.rerank_reports),
        elapsed_ms=first.elapsed_ms + second.elapsed_ms,
    )


def format_multi_result(result: MultiRetrievalResult) -> str:
    """把多查询结果格式化成给 LLM 读的文本（逐条 query 的命中情况 + 去重后的片段）。

    与 `_format_result` 共用同一套出处格式与上下文预算（`MAX_CHUNK_CHARS` / `MAX_TOTAL_CHARS`），
    预算按**合并后**的总量算——否则每条 query 各留 4000 字符，几条下来就把上下文撑爆了。
    """
    lines = [f"知识库检索结果｜共 {len(result.queries)} 条查询｜{result.summary()}"]
    for report in result.reports:
        mark = "✗ 低置信拒答" if report.low_confidence else "✓"
        lines.append(f"  {mark} {report.summary()}｜{report.query}")
    if result.refused_queries:
        lines.append(f"（上述 {len(result.refused_queries)} 条查询知识库中无相关内容，按约定未返回其片段）")

    # 重排前后各留一行：head_gap 是「重排到底有没有用」的量化答案，而且它是**相对指标**，
    # 不受两种分数量纲不同的影响（余弦 vs Sigmoid 后的 cross-encoder 分）。
    gain = result.head_gap_gain
    if gain:
        lines.append(f"  （重排收敛：head_gap 均值 {gain[0]:.3f} → {gain[1]:.3f}）")
    elif result.reranked:
        lines.append("  （已启用重排：片段按重排分排序，下方分数为**重排分**，与相似度不可比）")
    if not result.documents:
        lines.append(
            "\n请如实告知用户知识库中没有相关资料，并可以建议用户换个说法或补充背景信息；"
            "不要凭已有知识作答。"
        )
        return "\n".join(lines)

    score_label = "重排分" if result.reranked else "相似度"
    used = 0
    for index, doc in enumerate(result.documents, 1):
        text = doc.page_content
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + "……（本片段已截断）"
        block = (
            f"\n[{index}] {score_label} {_order_score(doc):.3f}"
            f"｜{_source_line(doc.metadata)}"
            f"\n    命中查询：{result.hit_by.get(_dedup_key(doc), '')}"
            f"\n{text}"
        )
        if used + len(block) > MAX_TOTAL_CHARS:
            # 不静默截断：明确说明还剩几条没展示
            lines.append(f"\n（其余 {len(result.documents) - index + 1} 条超出上下文预算，未展示）")
            break
        used += len(block)
        lines.append(block)

    return "\n".join(lines)


def search_knowledge_base_multi(
    queries: List[str],
    top_k: int = DEFAULT_TOP_K_PER_QUERY,
    use_rerank: bool = USE_RERANK,
) -> str:
    """多查询检索的文本入口：一组 query 进，去重合并后的片段文本出。

    与 `search_knowledge_base` 的区别只是「一次几条 query」；拒答、出处、上下文预算的
    规则完全一致。重写失败退化成单条 query 时也走这里（`queries` 长度为 1）。

    **不抛异常**：空列表、参数非法、检索失败一律转成 `"检索失败：..."` 文本返回。
    """
    if not [q for q in (queries or []) if (q or "").strip()]:
        return "检索失败：查询列表为空，请提供要检索的问题。"

    try:
        result = retrieve_multi(queries, k=top_k, use_rerank=use_rerank)
    except Exception as exc:                      # noqa: BLE001
        return f"检索失败：{type(exc).__name__}: {exc}"

    return format_multi_result(result)


# -------------------- LangChain tool --------------------
KNOWLEDGE_BASE_TOOL = StructuredTool.from_function(
    func=search_knowledge_base,
    name="search_knowledge_base",
    description=(
        "检索离心泵运维知识库，返回最相关的原文片段及其出处（文件名 / 章节 / 页码）。"
        "知识库内容：HCP 系列卧式单级单吸离心泵组的仿真操作与维修手册、"
        "工业离心泵标准操作维护规程（IOM）与故障排查知识库——"
        "涵盖设备结构与核心测点、启停与操作步骤、报警阈值、故障诊断与处理、维修记录等。"
        "凡涉及离心泵的操作、参数、报警、故障排查等问题，都应先调用本工具，"
        "依据检索到的原文作答，不要仅凭已有知识直接回答。"
        "答案中请标注片段出处。"
        "若返回内容提示「知识库中没有找到与问题相关的内容」，说明知识库确实没有资料，"
        "应如实告知用户，不要编造。"
    ),
)
