"""CentriPump-PHM 诊断流程的 LangGraph 拓扑。"""

from langgraph.graph import END, START, StateGraph

from src.orchestrator.nodes import (
    data_node,
    engineer_text_node,
    evidence_merge_node,
    manual_node,
    reasoner_node,
    reporter_node,
    router_node,
    safety_guard_node,
    # scada_fetch_node,
    vision_node,
)
from src.schemas.state import DiagnosisState


def build_diagnosis_graph():
    """构建诊断主图。

    Router 完成输入拆分后，工程师文本、SCADA 时序数据、图片数据三路
    独立处理；在证据合并后依次进入手册、推理、安全和报告阶段。

    Router 负责写入 route。SCADA 和 Vision 节点将来根据 route 自行跳过
    不存在的输入，因此图的拓扑保持稳定，证据合并节点也不会等待缺失分支。
    """
    graph = StateGraph(DiagnosisState)

    graph.add_node("router", router_node)
    graph.add_node("engineer_text", engineer_text_node)
    # graph.add_node("scada_fetch", scada_fetch_node)
    graph.add_node("data", data_node)
    graph.add_node("vision", vision_node)
    graph.add_node("evidence_merge", evidence_merge_node)
    graph.add_node("manual", manual_node)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("safety_guard", safety_guard_node)
    graph.add_node("reporter", reporter_node)

    graph.add_edge(START, "router")
    graph.add_edge("router", "engineer_text")
    # graph.add_edge("router", "scada_fetch")
    graph.add_edge("router", "vision")
    graph.add_edge("router", "data")
    graph.add_edge(
        ["engineer_text", "data", "vision"],
        "evidence_merge",
    )
    graph.add_edge("evidence_merge", "manual")
    graph.add_edge("manual", "reasoner")
    graph.add_edge("reasoner", "safety_guard")
    graph.add_edge("safety_guard", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()
