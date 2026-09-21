"""
CentriPump-PHM 诊断流程的 LangGraph 拓扑。

整体流程
--------
当前诊断流程采用固定拓扑，不根据输入动态修改 Graph 结构。

    START
      │
      ▼
  start_node
      │
      ├──────────────┬──────────────┐
      ▼              ▼              ▼
engineer_text      data           vision
      │              │              │
      └──────────────┴──────────────┘
                     │
                     ▼
               evidence_merge
                     │
                     ▼
                   manual
                     │
                     ▼
                  reasoner
                     │
                     ▼
                safety_guard
                     │
                     ▼
                  reporter
                     │
                     ▼
                    END


设计说明
--------
1. start_node 仅作为流程开始节点。

   当前不再承担条件路由职责，也不根据输入决定后续节点。

   它主要负责：
       - 记录诊断流程开始
       - 完成必要的初始状态准备
       - 为后续节点提供统一的流程入口

   因此当前 Graph 的拓扑是固定的。


2. 三个证据节点始终执行。

   engineer_text：
       负责处理工程师提供的文字、描述等工程信息。
       没有对应输入时，由节点自身记录 EMPTY。

   data：
       负责处理 SCADA、运行数据等结构化数据。
       没有对应输入时，由节点自身记录 EMPTY。

   vision：
       负责处理现场图片、视觉信息等。
       没有图片时，由节点自身记录 NO_IMAGE。

   因此：
       “没有输入” != “跳过节点”。

   节点始终存在并执行，由节点内部决定如何处理空输入。


3. 三个证据节点完成后，再进入 evidence_merge。

   evidence_merge 统一收集：

       engineer_text
       data
       vision

   三类证据，并形成后续诊断推理所需要的统一证据上下文。

   LangGraph 会等待三个上游节点全部完成后，
   才继续执行 evidence_merge。


4. 后续流程采用严格的固定顺序。

       evidence_merge
            ↓
          manual
            ↓
         reasoner
            ↓
       safety_guard
            ↓
         reporter

   各节点职责由对应节点自身负责，
   Graph 只负责定义执行顺序。


Checkpoint
----------
5. 当前阶段使用 SQLite 作为 LangGraph Checkpoint 存储。

   SqliteSaver 在本文件内部创建。

   采用模块级 Graph 单例：

       diagnosis_graph = build_diagnosis_graph()

   应用启动时创建一次，后续请求直接复用。

   不在每次请求中重新创建 Graph 或 Checkpoint。


6. Checkpoint 使用 thread_id 区分不同诊断任务。

   当前约定：

       trace_id == thread_id

   trace_id 由调用方（main / API）负责创建。

   Graph 本身不生成 trace_id。


7. 当前阶段采用同步 SqliteSaver。

   这是为了保持实现简单，优先完成诊断流程。

   当前不处理 AsyncSqliteSaver、
   异步 Checkpoint 生命周期等问题。

   后续如果需要完整异步化，再单独重构。


Graph 职责边界
--------------
8. 本文件只负责：

       - 创建 StateGraph
       - 注册诊断节点
       - 定义节点之间的拓扑关系
       - 创建 SQLite Checkpoint
       - 编译 Graph
       - 暴露全局 diagnosis_graph

   本文件不负责：

       - 生成 trace_id
       - 执行具体诊断逻辑
       - 修改业务节点内部行为
       - 编写 Prompt
       - 保存业务结果
       - 处理 API 请求


运行方式
--------
9. 外部调用方直接复用 diagnosis_graph：

       diagnosis_graph.invoke(
           state,
           config={
               "configurable": {
                   "thread_id": trace_id,
               }
           },
       )

   不需要每次请求重新构建 Graph。

"""

from pathlib import Path
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.core.logger import setup_logging
from src.orchestrator.nodes import (
    data_node,
    engineer_text_node,
    evidence_merge_node,
    manual_node,
    reasoner_node,
    reporter_node,
    start_node,
    safety_guard_node,
    vision_node,
)
from src.schemas.state import DiagnosisState


# =============================================================================
# 一、Checkpoint 配置
# =============================================================================

# 项目运行目录下的 checkpoint 数据库。
#
# 如果你希望放到统一数据库目录，可以改成：
#
#     E:/.../data/checkpoints/checkpoints.db
#
# 当前使用相对路径，便于开发阶段运行。
CHECKPOINT_DB = Path("data/checkpoints.db")

def build_diagnosis_graph():
    """
    构建并编译 CentriPump-PHM 主诊断图。

    Returns
    -------
    CompiledStateGraph
        已经绑定 SQLite checkpoint 的 LangGraph。
    """

    # 先初始化项目日志。节点本身不需要任何日志 wrapper：logger 会在记录时
    # 直接读取 LangGraph 当前 RunnableConfig 中的 node / namespace 信息。
    setup_logging()

    # -------------------------------------------------------------------------
    # 确保 checkpoint 数据库目录存在
    # -------------------------------------------------------------------------

    CHECKPOINT_DB.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph = StateGraph(DiagnosisState)

    # 注册节点
    graph.add_node("start_node", start_node)
    graph.add_node("engineer_text", engineer_text_node)
    graph.add_node("data", data_node)
    graph.add_node("vision", vision_node)
    graph.add_node("evidence_merge", evidence_merge_node)
    graph.add_node("manual", manual_node)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("safety_guard", safety_guard_node)
    graph.add_node("reporter", reporter_node)

    # START -> start
    graph.add_edge(START, "start_node")

    # start -> 三个证据节点
    graph.add_edge("start_node", "engineer_text")
    graph.add_edge("start_node", "data")
    graph.add_edge("start_node", "vision")

    # 三个证据节点 -> merge
    graph.add_edge(
        [
            "engineer_text",
            "data",
            "vision",
        ],
        "evidence_merge",
    )

    # 后续固定流程
    graph.add_edge("evidence_merge", "manual")
    graph.add_edge("manual", "reasoner")
    graph.add_edge("reasoner", "safety_guard")
    graph.add_edge("safety_guard", "reporter")
    graph.add_edge("reporter", END)

    # ---------------------------------------------------------
    # 创建长期存在的 SQLite connection
    # ---------------------------------------------------------

    conn = sqlite3.connect(
        CHECKPOINT_DB,
        check_same_thread=False,
    )

    # 创建 checkpoint saver
    checkpointer = SqliteSaver(conn)

    return graph.compile(
        checkpointer=checkpointer,
    )

# =============================================================================
# 二、全局 Graph
# =============================================================================

# 应用启动时创建一次。
#
# 不要每次请求都重新创建 Graph。
#
# 使用：
#
#     from src.orchestrator.graph import diagnosis_graph
#
# 然后：
#
#     diagnosis_graph.invoke(
#         state,
#         config={
#             "configurable": {
#                 "thread_id": trace_id,
#             }
#         },
#     )
#
diagnosis_graph = build_diagnosis_graph()
