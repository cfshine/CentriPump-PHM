# src/orchestrator/graph.py
"""LangGraph 状态图定义（节点拓扑注册、App 编译）。

本文件定义：

  1. Step 2 子图 —— 取数 → 确定性计算 → 语义化（``DataAgentState``）
  2. Step 3 子图 —— 读图 → OCR/视觉 → 固定信封（``VisionAgentState``）
  3. ``build_main_graph()`` —— 主图串行挂载 Step 2 → Step 3。

``image_refs`` 为空时 Step 3 ingest 直接空信封结束，不调模型。
"""
from langgraph.graph import StateGraph, START, END

from src.schemas.state import DiagnosisState
from src.sub_agents.data_agent.data_graph import data_agent_graph
from src.sub_agents.vision_agent.vision_graph import vision_agent_graph


def build_main_graph(*, checkpointer=None):
    """构建主图。主图状态是 ``DiagnosisState``（公共契约）。"""
    main = StateGraph(DiagnosisState)

    main.add_node("step2", data_agent_graph)
    main.add_node("step3", vision_agent_graph)

    main.add_edge(START, "step2")
    main.add_edge("step2", "step3")
    main.add_edge("step3", END)

    return main.compile(checkpointer=checkpointer)


__all__ = ["data_agent_graph", "vision_agent_graph", "build_main_graph"]
