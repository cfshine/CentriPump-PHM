"""
CentriPump-PHM 的 LangGraph 节点。

节点设计
--------
1. start_node节点
    更新:
        取消了原本的分流节点router_node, 数据实际情况交由各节点判断, 
        缺少必要参数, 无数据, 缺数据等状态同等记录
    功能:
        现在仅仅作为启动项, 比如日志的开始
"""

from src.schemas.state import DiagnosisState
from src.core.logger import get_logger


logger = get_logger(__name__)


# =============================================================================
# Step 1
# =============================================================================


def start_node(state: DiagnosisState) -> dict:
    logger.info("诊断流程开始")
    return {}


# =============================================================================
# Step 2
# =============================================================================


def engineer_text_node(state: DiagnosisState) -> dict:
    """
    Step 2：工程师文本处理。

    当前阶段暂时不接入 LLM。

    如果没有用户文本，则记录为空输入。

    后续正式实现时，可以在这里：

        context.user_query
            ↓
        Engineer Text Agent
            ↓
        engineer_text
    """

    if not state.context.user_query.strip():
        logger.info("未提供工程师文本，记录为空输入")
        # ---------------------------------------------------------------------
        # 当前 DiagnosisState 中暂时没有 engineer_text 专属区域。
        #
        # 所以当前阶段不新增 State 字段，只通过空返回表示：
        #
        #     没有工程师文本
        #
        # 等你正式确定 EngineerTextState 后，
        # 再写入：
        #
        #     engineer_text.status = "EMPTY"
        #
        # ---------------------------------------------------------------------
        return {}

    # TODO:
    # 后续在这里调用工程师文本 Agent。
    #
    # 例如：
    #
    # result = engineer_text_agent(...)
    #
    # return {
    #     "engineer_text": result
    # }

    logger.info("开始处理工程师文本")
    return {}


# =============================================================================
# Step 3
# =============================================================================


def data_node(state: DiagnosisState) -> dict:
    logger.info("数据处理开始")

    # 延迟导入：本模块会被主图导入，而子 Agent 会连带初始化大模型客户端
    from src.sub_agents.data_agent.data_graph import run_data_agent
    result = run_data_agent(state)

    logger.info("数据处理结束")

    return result


def vision_node(state: DiagnosisState) -> dict:

    """
    提取图片故障特征。
    参数：
        state: 全局状态；Step 3 读 ``state.context.image_refs`` 与 device_id / alarm_code。

    返回：
        ``{"vision": VisionState}`` —— 只回写 vision 一个盒子。
    """

    logger.info("视觉处理开始")

    # 延迟导入：同上，避免主图 import 本模块时就初始化大模型客户端
    from src.sub_agents.vision_agent.vision_nodes import vision_node as run_vision_agent
    result = run_vision_agent(state)

    logger.info("视觉处理结束")

    return result


def evidence_merge_node(state: DiagnosisState) -> dict:
    """
    合并工程师文本、时序数据和图片分析结果。

    这个节点不应该重新解释证据。

    它主要负责确认：

        当前有哪些证据源实际产生了结果。

    后续可以把这些结果整理成 Reasoner 所需要的统一输入。
    """

    # TODO:
    #
    # 正式实现：
    #
    #     engineer_text
    #     data
    #     vision
    #
    # 分别保留来源，不要把不同来源揉成一段无法追溯的文本。

    logger.info("开始合并多源证据")
    return {}


# =============================================================================
# Manual
# =============================================================================


def manual_node(state: DiagnosisState) -> dict:
    """
    Step 4：调用 Manual Agent 检索维修手册与规程。
    """

    # TODO:
    #
    # 根据：
    #
    #     data
    #     vision
    #     engineer_text
    #
    # 构造检索查询。
    #
    # 然后写入：
    #
    #     state.manual

    logger.info("开始检索维修手册")
    return {}


# =============================================================================
# Reasoner
# =============================================================================


def reasoner_node(state: DiagnosisState) -> dict:
    """
    Step 5：综合多源证据，生成故障机理与归因。
    """

    # TODO:
    #
    # 调用 Reasoner / LLM。
    #
    # 输出：
    #
    #     reasoning.root_cause
    #     reasoning.hypotheses
    #     reasoning.confidence
    #     reasoning.support_sources
    #     reasoning.summary
    #     reasoning.conflicts
    #     reasoning.used_inputs

    logger.info("开始综合证据并推理")
    return {}


# =============================================================================
# Safety Guard
# =============================================================================


def safety_guard_node(state: DiagnosisState) -> dict:
    """
    Step 6：执行确定性的 RAM 风险和人工转交判断。

    这里禁止直接让 LLM 决定：

        safety.decision
        safety.risk_level

    必须使用：

        rules/
        ram_matrix.json
        确定性 Python 逻辑

    产生最终结果。
    """

    # TODO:
    #
    # safety_result = safety_guard.evaluate(
    #     state
    # )
    #
    # return {
    #     "safety": safety_result
    # }

    logger.info("开始执行安全门禁")
    return {}


# =============================================================================
# Reporter
# =============================================================================


def reporter_node(state: DiagnosisState) -> dict:
    """
    Step 7：生成可溯源的最终排查工单。
    """

    # TODO:
    #
    # Reporter 应该根据：
    #
    #     reasoning
    #     safety
    #     manual
    #     data
    #     vision
    #
    # 生成最终工单。
    #
    # 完整工单正文不要塞进 State。
    #
    # State 只保存：
    #
    #     diagnosed_at
    #     report_ref
    #     report_uri
    #     traceability

    logger.info("开始生成诊断报告")
    return {}
