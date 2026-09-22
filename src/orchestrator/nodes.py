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

from datetime import datetime, timedelta

from src.schemas.state import (
    AlarmState,
    DataQuality,
    DataState,
    DiagnosisState,
    EngineerTextState,
    TelemetryRef,
    VisionState,
)
from src.core.logger import get_logger


logger = get_logger(__name__)


# =============================================================================
# Step 1
# =============================================================================


def start_node(state: DiagnosisState) -> dict:
    """
    Step 1：流程启动节点。
    完成基础环境检查与初始化，若未提供时间窗口，默认补充为当前时间往前半小时。
    """
    logger.info("诊断流程开始")
    ctx = state.context
    updates = {}

    # 检测时间窗口：如果不存在，默认从当前时间往前半个小时
    start_time = (ctx.start_time or "").strip()
    end_time = (ctx.end_time or "").strip()
    if not (start_time and end_time):
        now = datetime.now().replace(microsecond=0)
        default_start = (now - timedelta(minutes=30)).isoformat()
        default_end = now.isoformat()
        logger.info("检测到时间窗口不存在，设置默认时间窗口: %s ~ %s", default_start, default_end)
        updates["context"] = ctx.model_copy(
            update={"start_time": default_start, "end_time": default_end}
        )

    return updates


# =============================================================================
# Step 2
# =============================================================================


def engineer_text_node(state: DiagnosisState) -> dict:
    """
    Step 2：工程师文本处理。

    检测是否存在工程师描述：
    - 若不存在描述：更新 state 为无描述 (EMPTY)，并决定跳过后续文本分析；
    - 若存在描述：记录输入内容并更新 state 为 RECORDED。
    """
    query = (state.context.user_query or "").strip()
    if not query:
        logger.info("未检测到描述，记录为无描述 (EMPTY)，跳过后续文本分析")
        return {
            "engineer_text": EngineerTextState(
                status="EMPTY",
                text="",
                summary="无描述：未提供工程师描述文本",
            )
        }

    logger.info("检测到工程师描述 (长度: %d 字符)，开始处理", len(query))
    try:
        logger.info("工程师文本记录成功: %s", query[:50] + ("..." if len(query) > 50 else ""))
        logger.info("工程师文本处理完成")
        return {
            "engineer_text": EngineerTextState(
                status="RECORDED",
                text=query,
                summary=query,
            )
        }
    except Exception as exc:
        logger.exception("工程师文本处理异常: %s", exc)
        return {
            "engineer_text": EngineerTextState(
                status="EMPTY",
                text="",
                summary=f"工程师文本处理异常: {type(exc).__name__}",
            )
        }


# =============================================================================
# Step 3
# =============================================================================


def data_node(state: DiagnosisState) -> dict:
    """
    Step 3：时序数据分析。

    检测是否存在设备号与时间窗口：
    - 若不存在设备号：记录为无设备码 (EMPTY)，并决定跳过后续 SCADA 数据查询与分析；
    - 若时间窗口不存在：兜底使用当前时间往前半小时作为默认窗口；
    - 若设备号存在：调用 Data Agent 子图拉取 SCADA 时序数据并计算指标，回写 data 盒子。
    """
    ctx = state.context
    device_id = (ctx.device_id or "").strip()

    # 解析时间窗口（若缺失则兜底为当前时间往前半小时）
    start_time = (ctx.start_time or "").strip()
    end_time = (ctx.end_time or "").strip()
    if not (start_time and end_time):
        now = datetime.now().replace(microsecond=0)
        start_time = (now - timedelta(minutes=30)).isoformat()
        end_time = now.isoformat()
        logger.info("未提供时间窗口，使用默认时间窗口 (往前半小时): %s ~ %s", start_time, end_time)

    # 检测是否存在设备号
    if not device_id:
        logger.warning("未检测到设备号，记录为无设备码 (EMPTY)，跳过后续数据分析")
        telemetry_ref = TelemetryRef(
            source="scada_db.scada_telemetry",
            trace_id=ctx.trace_id,
            device_id="",
            start_time=start_time,
            end_time=end_time,
        )
        quality = DataQuality(
            status="EMPTY",
            total_points=0,
            reason="无设备码：未提供设备编号，跳过时序数据分析",
        )
        return {
            "data": DataState(
                telemetry_ref=telemetry_ref,
                quality=quality,
                metrics={},
                threshold_flags=[],
                alarms=AlarmState(effective="NONE", last="NONE", all=[]),
                descriptions=[],
            )
        }

    logger.info("时序数据分析开始: 设备=%s, 时间窗口=%s ~ %s", device_id, start_time, end_time)

    # 构造带有完整时间窗口的状态传入子图
    state_to_run = state
    if ctx.start_time != start_time or ctx.end_time != end_time:
        state_to_run = state.model_copy(
            update={"context": ctx.model_copy(update={"start_time": start_time, "end_time": end_time})}
        )

    try:
        # 延迟导入：本模块会被主图导入，而子 Agent 会连带初始化数据库及大模型客户端
        from src.sub_agents.data_agent.data_graph import run_data_agent

        result = run_data_agent(state_to_run)
    except Exception as exc:
        logger.exception("时序数据分析失败: %s", exc)
        telemetry_ref = TelemetryRef(
            source="scada_db.scada_telemetry",
            trace_id=ctx.trace_id,
            device_id=device_id,
            start_time=start_time,
            end_time=end_time,
        )
        quality = DataQuality(
            status="EMPTY",
            total_points=0,
            reason=f"时序数据处理异常降级: {type(exc).__name__}",
        )
        return {
            "data": DataState(
                telemetry_ref=telemetry_ref,
                quality=quality,
                metrics={},
                threshold_flags=[],
                alarms=AlarmState(effective="NONE", last="NONE", all=[]),
                descriptions=[],
            )
        }

    data = result.get("data")
    if data:
        quality_status = getattr(data.quality, "status", "UNKNOWN")
        flags_count = len(data.threshold_flags) if data.threshold_flags else 0
        desc_count = len(data.descriptions) if data.descriptions else 0
        logger.info(
            "时序数据分析完成: 质量状态=%s, 触发规则数=%d, 现象描述数=%d",
            quality_status,
            flags_count,
            desc_count,
        )
    else:
        logger.info("时序数据分析完成 (无返回数据)")

    return result


def vision_node(state: DiagnosisState) -> dict:
    """
    Step 3：现场图片故障特征分析。

    检测是否存在现场图片：
    - 若不存在图片：记录为无图片 (NO_IMAGE)，并决定跳过后续视觉大模型分析；
    - 若存在图片：调用 Vision Agent 逐图执行预处理、OCR 辅助与视觉大模型分析，回写 vision 盒子。
    """
    image_refs = state.context.image_refs or []
    if not image_refs:
        logger.info("未检测到图片，记录为无图片 (NO_IMAGE)，跳过后续视觉分析")
        return {
            "vision": VisionState(
                status="NO_IMAGE",
                findings=[],
                summary="无图片：未提供现场图片，跳过视觉分析",
            )
        }

    logger.info("检测到 %d 张图片，开始视觉分析", len(image_refs))
    try:
        # 延迟导入：避免主图 import 本模块时连带初始化大模型客户端
        from src.sub_agents.vision_agent.vision_nodes import vision_node as run_vision_agent

        result = run_vision_agent(state)
    except Exception as exc:
        logger.exception("视觉分析失败: %s", exc)
        return {
            "vision": VisionState(
                status="FAILED",
                findings=[],
                summary=f"视觉分析异常降级: {type(exc).__name__}",
            )
        }

    vision = result.get("vision")
    if vision:
        findings_count = len(vision.findings) if vision.findings else 0
        logger.info(
            "视觉分析完成: 状态=%s, 检出缺陷数=%d",
            vision.status,
            findings_count,
        )
    else:
        logger.info("视觉分析完成 (无返回数据)")

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
