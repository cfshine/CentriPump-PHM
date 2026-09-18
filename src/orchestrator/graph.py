# src/orchestrator/graph.py
"""LangGraph 状态图定义（节点拓扑注册、App 编译）。

本文件定义：

  1. Step 2 —— 子图：取数 → 确定性计算 → 语义化（``DataAgentState``，2 个私有字段）
  2. Step 3 —— **普通节点函数** ``vision_node``：读图 → OCR 辅助 → 视觉大模型
     （没有私有状态、只有线性流程，所以不包子图；2026-09-17 精简）
  3. ``build_main_graph()`` —— 主图串行挂载 Step 2 → Step 3。

两个节点都有"缺输入就安全退出"的行为，所以主图不需要额外的条件边：
  · ``image_refs`` 为空 → Step 3 一行日志直接返回；
  · 没有时间窗口   → Step 2 三个节点各自软降级（不查库、不算、不调大模型）。
"""
from langgraph.graph import StateGraph, START, END

from src.schemas.state import DiagnosisState
from src.sub_agents.data_agent.data_graph import data_agent_graph
from src.sub_agents.vision_agent.vision_nodes import vision_node


def build_main_graph(*, checkpointer=None):
    """构建主图。主图状态是 ``DiagnosisState``（公共契约）。

    参数：
        checkpointer: 可选的状态持久化器（默认 None = 不持久化）。
                      子图/节点都不需要自己的 checkpointer —— 挂在主图上时，
                      持久化由主图这一层统一负责。

    返回：
        编译好的主图（``CompiledStateGraph``），可直接 ``.invoke(state_dict)``。

    拓扑：
        ``START → step2 → step3 → END``（串行；两个节点互相不依赖，可改并行）
    """
    main = StateGraph(DiagnosisState)

    main.add_node("step2", data_agent_graph)
    main.add_node("step3", vision_node)

    main.add_edge(START, "step2")
    main.add_edge("step2", "step3")
    main.add_edge("step3", END)

    return main.compile(checkpointer=checkpointer)


__all__ = ["data_agent_graph", "vision_node", "build_main_graph"]
