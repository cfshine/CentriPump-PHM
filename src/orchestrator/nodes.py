"""CentriPump-PHM 的 LangGraph 节点占位定义。

节点只负责把 LangGraph State 适配到具体业务 Agent。当前阶段不接入
Agent、LLM、日志或持久化，因此函数体保持为空。
"""

from src.schemas.state import PumpDiagnosisState


def router_node(state: PumpDiagnosisState) -> dict:
    """提取实体并决定后续需要执行的诊断分支。"""
    pass


def engineer_text_node(state: PumpDiagnosisState) -> dict:
    """整理工程师输入的文本描述，作为独立证据来源。"""
    pass


def scada_fetch_node(state: PumpDiagnosisState) -> dict:
    """存在 machine_code 时，从 SCADA 获取设备时序数据。"""
    pass


def data_node(state: PumpDiagnosisState) -> dict:
    """存在 SCADA 数据时，调用 Data Agent 分析工业时序数据。"""
    pass


def vision_node(state: PumpDiagnosisState) -> dict:
    """存在 image_refs 时，调用 Vision Agent 提取图片故障特征。"""
    pass


def evidence_merge_node(state: PumpDiagnosisState) -> dict:
    """合并工程师文本、时序分析和图片分析，保留各自证据来源。"""
    pass


def manual_node(state: PumpDiagnosisState) -> dict:
    """调用 Manual Agent 检索维修手册与规程。"""
    pass


def reasoner_node(state: PumpDiagnosisState) -> dict:
    """综合多源证据，生成故障机理与归因。"""
    pass


def safety_guard_node(state: PumpDiagnosisState) -> dict:
    """执行确定性的 RAM 风险和人工转交判断。"""
    pass


def reporter_node(state: PumpDiagnosisState) -> dict:
    """生成可溯源的最终排查工单。"""
    pass
