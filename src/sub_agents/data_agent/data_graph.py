# src/sub_agents/data_agent/data_graph.py
"""Step 2 子图拓扑：``analyze``（取数+确定性计算） → ``summarize``（大模型语义化）。

★ 2026-09-20 适配组长的 9 盒子契约：
    · 子图 state 就是公共契约 ``DiagnosisState`` 本身（Step 2 没有任何私有字段）；
    · 节点只回写 ``data`` 一个顶层盒子，不再往顶层写扁平字段。

挂载方式（与组长约定为**路线 B**）：
    主图 ``src/orchestrator/nodes.py`` 里的 ``data_node`` 只是一行转调
    :func:`run_data_agent`，由它把子图结果回写给主图。
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

    ★ 不需要条件边的理由：两个节点都是"缺时间窗口就安全退出"，
      不是"走另一条分支"；条件边也解决不了"进入节点之后才失败"的中途异常。
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

    ★ 为什么只回写 data：
      主图里 data 与 vision 是**并行分支**（``router → data`` / ``router → vision``，
      最后汇合到 ``evidence_merge``）。若把整个 state 返回出去，就会把并行的 vision
      产出一起覆盖掉（同一个顶层键被两个并行节点写入）。

    ★ 为什么用 ``state.model_dump()`` 而不是直接把模型实例交给 ``invoke``：
      dict 输入一定被接受；模型实例是否被接受依赖 langgraph 的具体版本行为，
      这里不去依赖它（多一次浅序列化的开销可以忽略）。
    """
    out = data_agent_graph.invoke(state.model_dump())
    data = out["data"] if isinstance(out, dict) else out.data
    return {"data": data if isinstance(data, DataState) else DataState.model_validate(data)}
