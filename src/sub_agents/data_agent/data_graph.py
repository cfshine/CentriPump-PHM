from src.sub_agents.data_agent.data_state import DataAgentState
from .data_nodes import (
    calculate_metrics_node,
    fetch_data_node,
    semanticize_node,
)
from langgraph.graph import StateGraph, START, END

def build_data_agent_graph():
    """构建 Step 2 子图：fetch_data → calculate_metrics → semanticize。"""
    workflow = StateGraph(DataAgentState)

    workflow.add_node("fetch_data", fetch_data_node)
    workflow.add_node("calculate_metrics", calculate_metrics_node)
    workflow.add_node("semanticize", semanticize_node)

    workflow.add_edge(START, "fetch_data")
    workflow.add_edge("fetch_data", "calculate_metrics")
    workflow.add_edge("calculate_metrics", "semanticize")
    workflow.add_edge("semanticize", END)

    return workflow.compile()


#: 编译好的 Step 2 子图（模块级单例，供测试与主图复用）
data_agent_graph = build_data_agent_graph()