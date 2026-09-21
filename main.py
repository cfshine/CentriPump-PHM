"""导出当前 LangGraph 流程图。"""

from pathlib import Path
from src.orchestrator.graph import build_diagnosis_graph

DIAGNOSIS_GRAPH_PATH = Path("docs") / "diagnosis_graph.png"

def export_graph_png(output_path: Path = DIAGNOSIS_GRAPH_PATH) -> Path:
    """使用 LangGraph 自带的 Mermaid 渲染导出流程图。"""
    graph = build_diagnosis_graph()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph.get_graph().draw_mermaid_png(
        output_file_path=str(output_path),
    )
    return output_path


if __name__ == "__main__":
    print(export_graph_png())
