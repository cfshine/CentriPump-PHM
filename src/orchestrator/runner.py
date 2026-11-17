"""诊断图的唯一运行入口。

API/CLI 只在这里创建一次 trace_id；节点内部无需设置日志上下文。
"""

from __future__ import annotations

from typing import Any

from src.core.logger import get_logger, invoke_graph, log_context
from src.orchestrator.graph import diagnosis_graph
from src.schemas.state import DiagnosisState


logger = get_logger(__name__)


def run_diagnosis(state: DiagnosisState, *, trace_id: str | None = None,
                  report_path: str | None = None) -> Any:
    """运行一次诊断，默认令 LangGraph thread_id 与 trace_id 相同。"""

    resolved_trace_id = trace_id or state.context.trace_id
    if state.context.trace_id != resolved_trace_id:
        raise ValueError("state.context.trace_id 与入口 trace_id 必须一致")

    with log_context(
        trace_id=resolved_trace_id, thread_id=resolved_trace_id, graph_name="diagnosis"
    ):
        logger.info("提交诊断任务")
        try:
            result = invoke_graph(
                diagnosis_graph,
                state,
                trace_id=resolved_trace_id,
                thread_id=resolved_trace_id,
                graph_name="diagnosis",
            )
        except Exception:
            logger.exception("诊断任务失败")
            raise
        if report_path:
            from src.schemas.report import write_diagnosis_report
            write_diagnosis_report(DiagnosisState.model_validate(result), report_path)
        logger.info("诊断任务完成")
        return result
