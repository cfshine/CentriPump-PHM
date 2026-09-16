# src/orchestrator/graph.py
"""LangGraph 状态图定义（节点拓扑注册、App 编译）。

本文件定义两张图：

  1. ``build_data_agent_graph()`` —— Step 2 子图（取数 → 确定性计算 → 语义化）
     状态用 ``DataAgentState``：继承公共契约 + 补私有字段。

  2. ``build_main_graph()``       —— 主图，把 Step 2 子图作为一个节点挂上去。

父子图共享状态的做法（langgraph 1.2.11 实测确认）：

  · 子图 state **继承** ``DiagnosisState`` → 公共字段只定义一次，且子图能读到；
  · 子图独有的私有字段**不会**回流主图（父图 schema 里没有这些键）；
  · 因此挂载就是 ``add_node`` 一行，**不需要任何"翻译层"**。

Step 3~7 接入时照此办理：各自建子图，然后在 ``build_main_graph()`` 里加一行。
"""
from langgraph.graph import StateGraph, START, END

from src.schemas.state import DiagnosisState

from src.sub_agents.data_agent.data_graph import data_agent_graph


def build_main_graph(*, checkpointer=None):
    """构建主图。

    主图的状态是 ``DiagnosisState``（公共契约）；各 Step 以子图/节点的形式挂上来。
    当前只接了 Step 2 —— 其余 Step 接入时在这里加 ``add_node`` 与 ``add_edge``。
    """
    main = StateGraph(DiagnosisState)

    main.add_node("step2", data_agent_graph)

    main.add_edge(START, "step2")
    main.add_edge("step2", END)

    return main.compile(checkpointer=checkpointer)


__all__ = ["data_agent_graph", "build_main_graph"]
