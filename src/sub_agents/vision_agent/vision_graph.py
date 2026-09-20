"""Step 3 子图拓扑：``START → ingest →（有图）extract → END``。

为什么只有两个节点：
    真正的算法（OCR、体检、视觉理解、读数校正）都写在 ``analyze_one`` 这个普通函数里，
    图只负责"编排"——与 Step 2「算法在函数里、图只编排」保持同一风格。
    多图不是靠图上的循环边，而是 ``extract_node`` 内部一个 for 循环：
    LangGraph 的循环边会带来状态合并的复杂度，这里用不上。
"""

from langgraph.graph import END, START, StateGraph

from src.sub_agents.vision_agent.vision_nodes import extract_node, ingest_node
from src.sub_agents.vision_agent.vision_state import VisionAgentState


def _after_ingest(state: VisionAgentState) -> str:
    """条件边路由函数：ingest 之后决定"继续干活"还是"直接结束"。

    参数：
        state: 子图状态（只读 ``image_refs``）。

    返回：
        "extract" —— 有图片，进入 extract 节点开始处理。
        "end"     —— 没有图片，直接跳到 END（整条链路零成本：不读图、不 OCR、不调模型）。
    """
    if not state.get("image_refs"):
        return "end"
    return "extract"


def build_vision_agent_graph():
    """构建并编译 Step 3 子图。

    参数：
        无。

    返回：
        编译好的 LangGraph 应用（``CompiledStateGraph``），可直接 ``.invoke(state_dict)``；
        供 orchestrator 以 ``main.add_node("step3", vision_agent_graph)`` 挂到主图上。

    拓扑：
        START → ingest →（条件）→ extract → END
                       └（无图）→ END

    状态契约：
        用的是 ``VisionAgentState``（= 继承 ``DiagnosisState``，无私有字段）。
        子图能读到的键 = 自己在 schema 里声明的键；能回流父图的键 = 父图 schema 里也有的键。
        所以本子图只回写 ``visual_description`` 与 ``visual_findings`` 两个公共字段。
    """
    workflow = StateGraph(VisionAgentState)
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("extract", extract_node)
    workflow.add_edge(START, "ingest")
    workflow.add_conditional_edges(
        "ingest",
        _after_ingest,
        {"extract": "extract", "end": END},
    )
    workflow.add_edge("extract", END)
    return workflow.compile()


#: 编译好的子图单例（模块级）
vision_agent_graph = build_vision_agent_graph()