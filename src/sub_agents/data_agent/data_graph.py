# src/sub_agents/data_agent/data_graph.py
"""Step 2 子图拓扑：``analyze``（取数+确定性计算） → ``summarize``（大模型语义化）。
"""
from langgraph.graph import StateGraph, START, END

from src.schemas.state import DataState, DiagnosisState
from .data_nodes import analyze_node, summarize_node


def build_data_agent_graph():
    """构建 Step 2 子图。

    返回：
        编译好的子图（``CompiledStateGraph``），可直接 ``.invoke(state_dict)``，
        也可用 ``parent.add_node("data", data_agent_graph)`` 挂进主图。

    拓扑：
        ``START → analyze → summarize → END``（线性，无条件边）
    """
    workflow = StateGraph(DiagnosisState)

    workflow.add_node("analyze", analyze_node)
    workflow.add_node("summarize", summarize_node)

    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "summarize")
    workflow.add_edge("summarize", END)

    return workflow.compile()


#: 编译好的 Step 2 子图（模块级单例，供测试与主图复用）
data_agent_graph = build_data_agent_graph()


def run_data_agent(state: DiagnosisState) -> dict:
    """供主图 ``data_node`` 调用（挂载路线 B）：跑一遍 Step 2 子图，**只回写 data 盒子**。

    参数：
        state: 主图传入的全局状态（``DiagnosisState`` 实例）。

    返回：
        ``{"data": DataState}`` —— 只含 data 一个键。
    """
    out = data_agent_graph.invoke(state.model_dump())
    data = out["data"] if isinstance(out, dict) else out.data
    return {"data": data if isinstance(data, DataState) else DataState.model_validate(data)}
