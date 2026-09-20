"""CentriPump-PHM 的 LangGraph 节点占位定义。

节点只负责把 LangGraph State 适配到具体业务 Agent。
Step 2 / Step 3 的节点已接线（转调各自子 Agent）；其余节点仍是空占位。
"""

from src.schemas.state import DiagnosisState


def router_node(state: DiagnosisState) -> dict:
    """提取实体并决定后续需要执行的诊断分支。"""
    pass


def engineer_text_node(state: DiagnosisState) -> dict:
    """整理工程师输入的文本描述，作为独立证据来源。"""
    pass


# def scada_fetch_node(state: DiagnosisState) -> dict:
#     """存在 machine_code 时，从 SCADA 获取设备时序数据。"""
#     pass


def data_node(state: DiagnosisState) -> dict:
    """存在 SCADA 数据时，调用 Data Agent（Step 2）分析工业时序数据。

    参数：
        state: 全局状态；Step 2 只读 ``state.context``。

    返回：
        ``{"data": DataState}`` —— 只回写 data 一个盒子。
    """
    # 延迟导入：本模块会被主图导入，而子 Agent 会连带初始化大模型客户端
    from src.sub_agents.data_agent.data_graph import run_data_agent

    return run_data_agent(state)


def vision_node(state: DiagnosisState) -> dict:
    """存在 image_refs 时，调用 Vision Agent（Step 3）提取图片故障特征。

    参数：
        state: 全局状态；Step 3 读 ``state.context.image_refs`` 与 device_id / alarm_code。

    返回：
        ``{"vision": VisionState}`` —— 只回写 vision 一个盒子。
    """
    # 延迟导入：同上，避免主图 import 本模块时就初始化大模型客户端
    from src.sub_agents.vision_agent.vision_nodes import vision_node as run_vision_agent

    return run_vision_agent(state)


def evidence_merge_node(state: DiagnosisState) -> dict:
    """合并工程师文本、时序分析和图片分析，保留各自证据来源。"""
    pass


def manual_node(state: DiagnosisState) -> dict:
    """调用 Manual Agent 检索维修手册与规程。"""
    pass


def reasoner_node(state: DiagnosisState) -> dict:
    """综合多源证据，生成故障机理与归因。"""
    pass


def safety_guard_node(state: DiagnosisState) -> dict:
    """执行确定性的 RAM 风险和人工转交判断。"""
    pass


def reporter_node(state: DiagnosisState) -> dict:
    """生成可溯源的最终排查工单。"""
    pass
