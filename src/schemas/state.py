"""LangGraph 在离心泵诊断流程中传递的最小状态定义。"""

from typing import Any, TypedDict


class PumpDiagnosisState(TypedDict, total=False):
    """诊断流程各节点共享的状态。

    这里只固定流程骨架所需的字段；具体 Agent 的输入、输出和
    Pydantic Schema 将在各 Agent 落地时再逐步细化。
    """

    run_id: str
    request: dict[str, Any]
    engineer_text: str
    machine_code: str
    image_refs: list[str]
    route: dict[str, bool]
    scada_data: dict[str, Any]
    data_result: dict[str, Any]
    vision_result: dict[str, Any]
    evidence_package: dict[str, Any]
    manual_result: dict[str, Any]
    diagnosis_result: dict[str, Any]
    safety_result: dict[str, Any]
    report: dict[str, Any]
    error: str
