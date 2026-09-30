"""知识库：建库 / 向量化入库 / 检索。

文档加载与预处理（清洗 / 结构化 / 切片）在 doc_process.py，本模块只负责
「拿到待入库的文档块之后」的事情；面向 LLM 的检索入口在 retriever.py。两种用法：

    # 用法 1：全链路一条命令（内部按需 import doc_process，串起加载 → 切片 → 入库）
    from knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    kb.build("docs")              # 增量构建，幂等
    kb.build("docs", reset=True)  # 先清空 collection 再重建

    # 用法 2：逐阶段自己串（调试预处理规则时用，能单独重跑任意一段）
    from doc_process import (
        load_documents, clean_documents, structure_documents, split_documents,
    )

    documents = load_documents("docs")
    documents = clean_documents(documents)
    records   = structure_documents(documents)
    chunks    = split_documents(records)
    KnowledgeBase().vectorize_documents(chunks)
"""

import json
import os
import time

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # 只为类型注解：运行时不 import doc_process，保持模块单向依赖
    from doc_process import Chunk

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量

EMBEDDING_MODEL_DIR = os.getenv("EMBEDDING_MODEL_DIR")
RERANKER_MODEL_DIR = os.getenv("RERANKER_MODEL_DIR")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")

# 重排精度。实测（bge-reranker-v2-m3 + RTX 4060，30 对候选、正文平均 316 字符）：
#   fp32 / max_length=512 → 37.5 ms/对
#   fp16 / max_length=512 → 10.6 ms/对    ← 快 3.5 倍，且**分数完全一致**
#                                        （top1 0.938、中位 0.307 两者逐位相同）
# 因此默认「有 CUDA 就用 fp16」。CPU 上 fp16 反而更慢（无向量化加速），所以不是无条件开。
# max_length 实测 256/384/512 差异很小（本语料 chunk 平均 316 字符），维持 512。
RERANKER_FP16 = os.getenv("RERANKER_FP16", "auto").strip().lower()
_vector_store = None

# 向量库配置：持久化目录与集合名
VECTOR_DB_DIR = Path(__file__).resolve().parent / "chroma_db"
VECTOR_COLLECTION = "knowledge_base"
# 单次写入向量库的文档条数上限，避免一次请求过大
WRITE_BATCH_SIZE = 256

# 检索置信度下限（余弦相似度）。低于它就认为「知识库没有相关内容」，
# 应当触发「拒答 / 转人工」而不是硬塞给 LLM 生成——这是控制生成质量的关键闸门。
# 取值依据见 RETRIEVAL_LOW_CONFIDENCE 的实测注释（settings 见设计文档）。
RETRIEVAL_LOW_CONFIDENCE = 0.5


class RetrievalReport(BaseModel):
    """一次检索的质量评估。

    为什么检索完就要评估：检索质量是整条 RAG 链路的**上限**——rerank 只能在召回的候选里
    重新排序，LLM 只能基于喂进去的上下文作答。没有评估就无法回答「加了重排到底有没有变好」
    「这次检索该不该拒答」。

    评估指标分三类，对应三个用途：

    | 指标 | 用途 |
    | :-- | :-- |
    | `top1` / `mean` / `lowest` | 绝对质量；`lowest` 决定「最差的那条会不会拖累生成」 |
    | `head_gap`（top1 - top2） | 头部是否突出。rerank 前后的对比主要看它：重排正确时 gap 应变大 |
    | `distinct_sources` | 上下文是否被同一份资料刷屏——重复内容白占 LLM 的上下文预算 |
    | `low_confidence` | `top1 < threshold` → 知识库大概率没有相关内容，应拒答而非硬生成 |

    `score_key` 指明用哪个字段算分，因此**同一套评估既能评向量召回，也能评重排结果**
    （重排后传 `score_key="rerank_score"`）：注意两种分数量纲不同，跨阶段只比 `head_gap`
    这类「相对的」指标，别去比 `lowest` 的绝对值。
    """

    query: str = ""
    returned: int = 0
    score_key: str = "retrieval_score"
    top1: float = 0.0
    mean: float = 0.0
    lowest: float = 0.0
    head_gap: float = 0.0        # top1 - top2；只有一个结果时为 0
    distinct_sources: int = 0
    threshold: float = RETRIEVAL_LOW_CONFIDENCE
    low_confidence: bool = False
    elapsed_ms: float = 0.0

    def summary(self) -> str:
        """一行摘要，便于打日志与对比。"""
        return (
            f"{self.returned} 条 | {self.score_key} top1={self.top1:.3f} "
            f"均值={self.mean:.3f} 最低={self.lowest:.3f} 头距={self.head_gap:.3f} | "
            f"来源 {self.distinct_sources} 个 | {self.elapsed_ms:.0f}ms"
            + ("  ⚠️ 低置信" if self.low_confidence else "")
        )


class BuildReport(BaseModel):
    """一次知识库构建的结果（设计文档 4.6）。

    链路越长「静默失败」越危险：任何被跳过的文件、失败的解析，都必须在
    `skipped` / `failed` 里查得到，否则资料少了也没人知道。

    计数字段给了 0 默认值，`build` 可以在串联过程中逐段填写；
    跳过/失败原因用 `append` 累积。
    """

    files_scanned: int = 0
    docs_loaded: int = 0
    chunks_created: int = 0
    chunks_written: int = 0
    skipped: List[str] = Field(default_factory=list)   # 跳过的文件及原因
    failed: List[str] = Field(default_factory=list)    # 失败的文件及异常
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        """一行摘要，便于打日志（与 `RetrievalReport.summary` 同一套路子）。

        ⚠️ `docs_loaded` 是 Document 数，粒度是「PDF 一页 / Excel 一 Sheet /
        Markdown 整个文件」，与 `files_scanned`（原始文件数）不是一回事，
        实测 2 个文件能产出 24 个 Document——两个数分开报才不会被看串。
        """
        return (
            f"文件 {self.files_scanned} 个 -> Document {self.docs_loaded} 个 -> "
            f"chunk {self.chunks_created} 个（写入 {self.chunks_written} 条）| "
            f"跳过 {len(self.skipped)} 失败 {len(self.failed)} | {self.elapsed_sec:.2f}s"
        )


# 知识库本体：一个持久化向量知识库的建立 / 写入 / 读取 / 清空
# （不负责文本预处理，见 doc_process.py；不负责面向 LLM 的呈现，见 retriever.py）
class KnowledgeBase:
    def __init__(self, embedding: Optional[HuggingFaceEmbeddings] = None):
        self._embedding = embedding
        self._reranker = None
        self._validate_config()

    # -------------------- 配置校验 --------------------
    def _validate_config(self):
        if EMBEDDING_MODEL_DIR is None:
            raise ValueError("本地embedding模型未配置。")

    # -------------------- 向量化 --------------------
    @property
    def embeddings(self):
        if self._embedding is None:
            self._embedding = HuggingFaceEmbeddings(
                model_name = EMBEDDING_MODEL_DIR,
                model_kwargs = {"device": EMBEDDING_DEVICE},
                encode_kwargs = {"normalize_embeddings": True}
        )
        return self._embedding

    def chunk_to_document(self, chunk: "Chunk") -> Document:
        """把一个 `Chunk` 转成待入库的 `Document`——即设计文档 5.5 的入库转换层。

        `split_documents` 产出的是 `Chunk`（pydantic 契约），而写入侧要的是
        `Document`，中间这层转换是必须的、不能省。它做三件事：

        1. **正文**：`chunk.text` → `page_content`；
        2. **metadata 拍平**：`section_path`（list）拼成 `" > "` 字符串，
           `locator`（`{"page": 3}` / `{"sheet_name": …}`）展开成顶层键；
        3. **留溯源信息**：`chunk_id` / `doc_id` 一并写进去，检索返回的 Document
           才能直接说清「这句话出自哪份资料的哪个块」。

        注意这里**不做** id 生成：入库 id 直接用 `chunk.chunk_id`，由 doc_process
        按内容算好（见 `vectorize_documents`）。
        """
        metadata = {
            "source": chunk.source,
            "file_type": chunk.file_type,
            "section_path": " > ".join(chunk.section_path),
            "content_type": chunk.content_type,
            "char_count": chunk.char_count,
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            **chunk.locator,          # {"page": 3} / {"sheet_name": ...} -> 顶层键
        }
        return Document(page_content=chunk.text, metadata=metadata)

    def vectorize_documents(self, chunks: List["Chunk"]):
        """把文档块向量化并写入 Chroma，返回向量库实例。

        - **id 直接用 `chunk.chunk_id`**（doc_process 按内容算好的确定性 id），
          写入走 upsert，因此同一份资料重复构建只覆盖不新增（幂等）。写入侧**不再
          自己算一遍哈希**——两套 id 口径一旦不一致，幂等会静默失效。
        - 分批写入（每批 `WRITE_BATCH_SIZE` 条），避免单次请求过大。
        - 向量化由 `add_documents` 内部经 `embedding_function` 完成，不显式调用
          `embeddings.embed_documents()`；更细的批大小由 `encode_kwargs` 决定。
        - metadata 用 `chunk_to_document` 拍平后再过一遍 `_sanitize_metadata`
          兜底（实测 chromadb 1.5.9 只有 dict 会硬报错，压成标量是为了让同一
          字段在所有记录里形状一致）。

        依赖：pip install langchain-chroma
        """
        if not chunks:
            print("[WARN] 待向量化的文档块为空，跳过写入")
            return self.get_vector_store()

        store = self.get_vector_store()
        total = len(chunks)

        for start in range(0, total, WRITE_BATCH_SIZE):
            batch = chunks[start : start + WRITE_BATCH_SIZE]
            store.add_documents(
                documents=[
                    Document(
                        page_content=document.page_content,
                        metadata=self._sanitize_metadata(document.metadata),
                    )
                    for document in (self.chunk_to_document(chunk) for chunk in batch)
                ],
                ids=[chunk.chunk_id for chunk in batch],
            )
            print(f"[INFO] 已写入向量库 {min(start + WRITE_BATCH_SIZE, total)}/{total}")

        print(f"[INFO] 向量化完成：{VECTOR_DB_DIR} -> collection「{VECTOR_COLLECTION}」")
        return store

    @staticmethod
    def _sanitize_metadata(metadata: Optional[dict]) -> dict:
        """把 metadata 压成 Chroma 支持的标量字典（list/dict/None 均需转换）。"""
        cleaned = {}
        for key, value in (metadata or {}).items():
            if value is None:
                cleaned[key] = ""
            elif isinstance(value, (list, tuple)):
                cleaned[key] = " > ".join(str(item) for item in value)
            elif isinstance(value, dict):
                cleaned[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        # 实测 1.5.9 能接受空 metadata；这里仍兜底，保证入库记录形状统一
        return cleaned or {"source": ""}

    # -------------------- 获取向量数据库 --------------------


    def get_vector_store(self):
        """获取向量数据库（懒加载单例）

        内存里已有实例就直接复用；否则打开（不存在则新建）持久化目录下的集合。
        构建与检索共用同一个实例。langchain-chroma 指定 persist_directory 后
        每次写入自动落盘，无需再调用 persist()（该方法在 1.x 已移除）。
        """
        global _vector_store
        
        if _vector_store is None:
            try:
                from langchain_chroma import Chroma
            except ImportError as exc:
                raise ImportError(
                    "向量库需要 langchain-chroma，请先执行：pip install langchain-chroma"
                ) from exc

            VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
            _vector_store = Chroma(
                collection_name=VECTOR_COLLECTION,
                embedding_function=self.embeddings,
                persist_directory=str(VECTOR_DB_DIR),
            )
           
            print(f"[INFO] 向量库已就绪：{VECTOR_DB_DIR} -> collection「{VECTOR_COLLECTION}」")

        return _vector_store

    # -------------------- 检索 --------------------
    def retrieve(self, query: str, k: int = 5, min_score: Optional[float] = None) -> List[Document]:
        """检索知识库，返回与 query 最相关的 k 个 Document。

        - 每条结果带 `metadata["retrieval_score"]`（**余弦相似度**，1 = 完全一致）
          与 `metadata["retrieval_rank"]`（0 起的名次），既便于排查，也是后续
          rerank / LLM 判断可信度的依据；
        - 检索完立刻做一次质量评估（见 `evaluate_retrieval`）并打一行日志；
        - `min_score` 可先滤掉明显不相关的命中（如 0.45），避免它们占满上下文预算。

        **两阶段用法**（先多召回、再精排，`k` 就是召回量）：

            candidates = builder.retrieve(query, k=20)          # 多召回
            top5       = builder.rerank(query, candidates, top_n=5)   # 精排收敛

        `rerank` 会保留 metadata 并追加 `rerank_score`，所以重排后仍能拿
        `evaluate_retrieval(top5, score_key="rerank_score")` 对比前后质量。

        注意：入库侧 embedding 已做 L2 归一化（见 embeddings 属性），查询侧必须用
        同一模型、同样的归一化设置，否则相似度不可比。query 侧是否加指令前缀，
        取决于所用模型（bge-m3 一般不加），且必须与入库侧口径一致。
        """
        if not query.strip():
            print("[WARN] 查询为空，返回空结果")
            return []

        store = self.get_vector_store()
        started = time.perf_counter()
        hits = store.similarity_search_with_score(query, k=k)

        documents: List[Document] = []
        for rank, (doc, distance) in enumerate(hits):
            score = self._distance_to_cosine(distance)
            if min_score is not None and score < min_score:
                continue
            documents.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "retrieval_score": score,
                        "retrieval_rank": rank,
                    },
                )
            )

        report = self.evaluate_retrieval(
            documents,
            query=query,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
        print(f"[INFO] 检索「{query[:20]}」: {report.summary()}")
        if report.low_confidence:
            print(
                f"[WARN] top1={report.top1:.3f} < 阈值 {report.threshold}："
                "知识库可能没有相关内容，建议拒答或转人工，不要让 LLM 硬生成"
            )
        return documents

    def evaluate_retrieval(
        self,
        documents: List[Document],
        query: str = "",
        score_key: str = "retrieval_score",
        threshold: float = RETRIEVAL_LOW_CONFIDENCE,
        elapsed_ms: float = 0.0,
    ) -> RetrievalReport:
        """评估一批检索结果的质量（`RetrievalReport` 的字段含义见其 docstring）。

        只读入参、不改动它，因此可以随时对同一批结果重复调用；也正因为它认的是
        `score_key` 指定的字段，**同一函数既能评向量召回（`retrieval_score`）、
        也能评重排结果（`rerank_score`）**，这让「加了重排到底有没有变好」可量化。
        """
        scores = [
            float(doc.metadata[score_key])
            for doc in documents
            if score_key in doc.metadata
        ]
        if not scores:
            return RetrievalReport(
                query=query, score_key=score_key, threshold=threshold, elapsed_ms=elapsed_ms
            )

        top1 = max(scores)
        mean = sum(scores) / len(scores)
        return RetrievalReport(
            query=query,
            returned=len(documents),
            score_key=score_key,
            top1=top1,
            mean=mean,
            lowest=min(scores),
            head_gap=(top1 - sorted(scores)[-2]) if len(scores) > 1 else 0.0,
            distinct_sources=len({doc.metadata.get("source", "") for doc in documents}),
            threshold=threshold,
            low_confidence=top1 < threshold,
            elapsed_ms=elapsed_ms,
        )

    @staticmethod
    def _distance_to_cosine(distance: float) -> float:
        """把 Chroma 的 `l2` 距离换算成余弦相似度。

        ⚠️ 实测（chromadb 1.5.9 + 归一化的 bge-m3 向量）：**Chroma 的 l2 距离是「平方」欧氏距离**。
        对单位向量有 `d = ‖a-b‖² = 2 - 2·cos`，故 `cos = 1 - d/2`。

        本次实测对照：同一条记录 Chroma 返回 `0.203561`，手工算 L2² 也是 `0.203561`，
        而 L2 是 `0.451177` —— 说明确实是平方值。

        因此**不要用 `similarity_search_with_relevance_scores`**：它按 `1 - d/√2` 换算，
        那是「非平方」欧氏距离的公式，在本库上会系统性偏小（同一对向量：它给 0.856，
        正确余弦是 0.898）。分数一旦偏，后面 rerank 的对比、LLM 的置信度判断都会跟着错。
        """
        return max(-1.0, min(1.0, 1.0 - distance / 2.0))

    # -------------------- 重排序 --------------------
    @property
    def reranker(self):
        """懒加载 CrossEncoder 重排模型（进程内单例，与 embeddings 同一套路子）。"""
        if self._reranker is None:
            if RERANKER_MODEL_DIR is None:
                raise ValueError(
                    "重排模型未配置。请在 .env 中设置 RERANKER_MODEL_DIR："
                    "本地模型目录（如 /home/bicouper/models/BAAI/bge-reranker-v2-m3），"
                    "或 HuggingFace 模型名（如 BAAI/bge-reranker-v2-m3）。"
                )
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "重排序需要 sentence-transformers，请先执行：pip install sentence-transformers"
                ) from exc

            model_kwargs = {}
            if self._use_fp16():
                import torch  # 只在真要 fp16 时才 import，保持模块导入轻量
                model_kwargs["torch_dtype"] = torch.float16

            self._reranker = CrossEncoder(
                model_name_or_path=RERANKER_MODEL_DIR,
                max_length=512,
                device=RERANKER_DEVICE,
                model_kwargs=model_kwargs or None,
            )
            print(
                f"[INFO] 重排模型已加载：{RERANKER_MODEL_DIR}"
                f"（device={RERANKER_DEVICE}, "
                f"dtype={'float16' if model_kwargs else 'float32'}）"
            )
            # 预热一次：首次前向含 CUDA kernel 编译 / 显存分配，实测冷启动 32ms/对、
            # 预热后 10.6ms/对（3 倍差距）。放在加载路径里，让**第一次真实查询**不为
            # 一次性初始化买单——与 `get_vector_store()` 在计时前先触发的口径一致。
            self._reranker.predict([("预热", "预热")])
        return self._reranker

    @staticmethod
    def _use_fp16() -> bool:
        """`RERANKER_FP16` 的三种取值：`auto`（默认，有 CUDA 才开）/ `true` / `false`。"""
        if RERANKER_FP16 in ("true", "1", "yes", "on"):
            return True
        if RERANKER_FP16 in ("false", "0", "no", "off"):
            return False
        return RERANKER_DEVICE.startswith("cuda")

    def rerank(self, query: str, documents: List[Document], top_n: Optional[int] = None) -> List[Document]:
        """按与 query 的相关性对检索结果重排，返回新的 Document 列表。

        为什么需要：向量检索比较的是 query 向量与 chunk 向量的余弦相似度，粒度粗，
        容易被高频词带偏；重排是 cross-encoder 对 (query, chunk) 成对打分，精度更高，
        通常放在召回之后做二阶段精排——先多召回、再收敛。

        - **不修改入参**：返回的是新建的 Document，原列表顺序与对象都不动；
        - 分数写进 `metadata["rerank_score"]`（float），便于展示、过滤与排查；
        - `top_n=None` 时返回全部，只是顺序变了；同分保持原相对顺序（稳定排序）。

        ⚠️ **`rerank_score` 的语义与 `retrieval_score` 不同，两者不可比**（已实测）：

        | | `retrieval_score` | `rerank_score` |
        | :-- | :-- | :-- |
        | 来源 | 向量余弦 | cross-encoder 输出 |
        | 取值范围 | `[-1, 1]` | `[0, 1]` |
        | 激活 | 无（本身就是余弦） | **Sigmoid**（`sentence_transformers` 在 `num_labels=1` 时默认套上） |

        重排分**看起来**像概率、也在 `[0,1]` 内，于是容易被误当成「和余弦一个量纲、
        可以直接套 `RETRIEVAL_LOW_CONFIDENCE = 0.5`」——**不要这么做**。实测一条真实相关
        查询的 55 条候选：重排分最高 0.938、**均值仅 0.208**、最低 0.000，55 条全部
        `>= 0` 但绝大多数低于 0.5。拿 0.5 当闸门会把几乎所有候选拒掉——**明明有资料却拒答**。

        结论：**余弦管拒答闸门（阈值 0.5 已标定），重排分只管排序与截断**。
        底层的 logit 重排分（未过 sigmoid 的原始值）不是这个字段；要改激活函数，
        得先重新标定一套阈值，别只改代码。

        依赖：pip install sentence-transformers + .env 中的 `RERANKER_MODEL_DIR`
        """
        if not documents:
            return []

        scores = self.reranker.predict(
            [(query, doc.page_content) for doc in documents]
        )

        ranked = sorted(
            zip(documents, scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        if top_n is not None:
            ranked = ranked[:top_n]

        return [
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": float(score)},
            )
            for doc, score in ranked
        ]

    # -------------------- 全量重建 --------------------
    def reset_vector_store(self) -> None:
        """清空 `VECTOR_COLLECTION` 的全部记录，供全量重建使用（决策点 11）。

        **什么时候必须清**：id 由内容决定（4.4），所以切哈希口径、改清洗/切片参数之后，
        旧记录算出的 id 与新记录对不上 → upsert 覆盖不到它们，结果是**新旧两版内容同时
        留在库里、同时被检索命中**，且不报任何错。这时只能先清再建。

        实现上删「collection」而不是删 `chroma_db/` 目录，理由有三：

        1. 只清 `knowledge_base` 这一个集合，不波及同库里的其它 collection；
        2. 不用关 Chroma client / 释放文件句柄，在 WSL 挂载盘这类跨文件系统场景下
           删目录容易碰到文件占用报错；`reset_collection()` 走的是库内的元数据操作，
           不碰文件系统；
        3. 删完同进程内立刻可写，`_vector_store` 单例不用重建。

        注意这里走的是 `delete_collection()` + 重新建同名空集合（`reset_collection()`
        的内部实现），因此**只影响向量库，不影响源文件**——源资料还在，随时能重建回来。
        """
        store = self.get_vector_store()
        store.reset_collection()
        print(f"[INFO] 已清空向量库 collection「{VECTOR_COLLECTION}」：{VECTOR_DB_DIR}")

    # -------------------- 编排入口 --------------------
    def build(self, path: Union[str, Path], reset: bool = False) -> BuildReport:
        """串起全链路：加载 → 清洗 → 结构化 → 切片 → 向量化入库，返回构建报告。

        `path` 可以是单个文件，也可以是目录（递归）。幂等性由确定性 id + upsert 保证：
        **同一份资料重复构建条数不变**，只是把相同 id 的记录重写一遍。因此日常更新资料
        直接重跑即可，不需要 `reset`。

        `reset=True` 会**先清空整个 collection 再写入**（见 `reset_vector_store`）。
        用在「改了哈希口径 / 改了清洗切片参数，旧记录再也覆盖不到」的时候；
        代价是：本次没切出任何内容就不会清库（见下），以及**库里原有的其它资料会一并消失**
        ——所以 `reset=True` 的前提是 `path` 覆盖了知识库的全部来源，而不是只指向新增的那几份。

        每个阶段都落在具名变量上、阶段之间不共享可变状态，所以任何一段都能单独重跑、
        单独测（这也是为什么预处理那四个函数留在 `doc_process`，没有被塞进类里）。

        ⚠️ **reset 放在写入之前、而不是最开头**：加载或预处理一旦抛异常（路径写错、
        文件损坏），库还是原样，不会被「先清空、再失败」搞成空的。同理，本次切出 0 个
        chunk 时也跳过清库——多半是传错了路径，清空只会让人以为「资料丢了」。

        失败策略与 `load_documents` 一致：**单文件**读取失败只记进 `report.failed` 不中断
        整批；目录不存在则直接抛 `FileNotFoundError`；写入阶段失败会向上抛（不吞异常，
        宁可让调用方看到，也不要静默产出一个少了半截的库）。
        """
        started = time.perf_counter()

        # doc_process 只在这里按需 import：模块间保持单向依赖（见文件头），
        # 也让「不装 torch / chromadb」的环境仍能单独跑预处理那一段。
        from doc_process import (
            LoadStats,
            clean_documents,
            load_documents,
            split_documents,
            structure_documents,
        )

        stats = LoadStats()                                  # 加载统计的出参
        documents = load_documents(path, stats=stats)        # 1. 加载
        documents = clean_documents(documents)               # 2. 清洗（跨文档，去页眉页脚）
        records = structure_documents(documents)             # 3. 结构化 -> List[DocRecord]
        chunks = split_documents(records)                    # 4. 切片 -> List[Chunk]

        if reset and chunks:
            self.reset_vector_store()
        elif reset:
            print(
                "[WARN] reset=True 但本次没切出任何 chunk，已跳过清库——"
                "否则会先清空、再写不回去（检查 path 是否指对了资料目录）"
            )

        self.vectorize_documents(chunks)                     # 5. 向量化 + 入库（内部转换）

        report = BuildReport(
            files_scanned=stats.files_scanned,
            docs_loaded=stats.docs_loaded,
            chunks_created=len(chunks),
            chunks_written=len(chunks),   # 全部 chunk 都经 add_documents 提交；空输入时为 0
            skipped=list(stats.skipped),
            failed=list(stats.failed),
            elapsed_sec=time.perf_counter() - started,
        )
        print(f"[INFO] 知识库构建完成：{report.summary()}")
        if report.failed:
            print(f"[WARN] 有 {len(report.failed)} 份资料读取失败，未进入知识库：{report.failed}")
        return report


