"""日志上下文只依赖公开的节点运行配置，不需要节点 wrapper。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.core.logger import build_graph_config, get_log_context


def test_langgraph_node_context_is_available_without_wrapper() -> None:
    seen = []

    def node(state: dict) -> dict:
        seen.append(get_log_context())
        return state

    graph = StateGraph(dict)
    graph.add_node("vision", node)
    graph.add_edge(START, "vision")
    graph.add_edge("vision", END)
    graph.compile().invoke(
        {},
        config=build_graph_config(trace_id="trace-123", thread_id="thread-123"),
    )

    context = seen[0]
    assert context.trace_id == "trace-123"
    assert context.thread_id == "thread-123"
    assert context.node == "vision"
    assert context.graph_path == "diagnosis/vision"
