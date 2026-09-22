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

    # 若已有分析结果（非 PENDING/EMPTY），直接复用，避免重复分析或覆盖外部输入
    if getattr(state.data.quality, "status", None) in ("OK", "PARTIAL") and (
        state.data.threshold_flags or state.data.metrics or state.data.descriptions
    ):
        logger.info("已有时序数据分析结果，直接复用: 质量=%s, 触发规则数=%d", state.data.quality.status, len(state.data.threshold_flags))
        return {"data": state.data}

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
    # 若已有视觉分析结果（非 PENDING/NO_IMAGE），直接复用
    if getattr(state.vision, "status", None) in ("DEFECT_FOUND", "NO_DEFECT"):
        logger.info("已有视觉分析结果，直接复用: 状态=%s, 缺陷数=%d", state.vision.status, len(state.vision.findings))
        return {"vision": state.vision}

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
    证据汇聚节点。
    确认各前置证据源（工程师文本、SCADA时序数据、现场视觉分析）的产出情况，
    汇总证据状态并为后续手册检索与推理提供统一上下文。
    """
    sources_summary = []
    if getattr(state.engineer_text, "status", None) == "RECORDED":
        sources_summary.append("工程师现场描述")
    if getattr(state.data.quality, "status", None) in ("OK", "PARTIAL"):
        sources_summary.append(f"时序数据({state.data.quality.status}, {state.data.quality.total_points}点)")
    if getattr(state.vision, "status", None) in ("DEFECT_FOUND", "NO_DEFECT"):
        sources_summary.append(f"视觉分析({state.vision.status}, {len(state.vision.findings)}项发现)")

    logger.info("多源证据汇聚完成，有效证据源: %s", ", ".join(sources_summary) if sources_summary else "无")
    return {}


# =============================================================================
# Manual
# =============================================================================


def manual_node(state: DiagnosisState) -> dict:
    """
    Step 4：手册 / RAG 知识检索节点（当前阶段由高拟真硬编码规程引擎支撑）。
    基于设备类型、时序告警码、阈值特征与视觉缺陷，检索对应的《离心泵使用维护说明书》(IOM)及国标条款。
    """
    from src.schemas.state import (
        ManualEvidence,
        ManualState,
        ProcedureRequirement,
        RecommendedAction,
    )

    logger.info("开始检索维修手册与工程规程 (硬编码RAG模块)")
    flags = set(state.data.threshold_flags or [])
    alarm = (state.context.alarm_code or "").upper()
    user_query = (state.context.user_query or "").lower()
    vision_defect_types = {f.defect_type.upper() for f in (state.vision.findings or [])}

    evidences: list[ManualEvidence] = []
    requirements: list[ProcedureRequirement] = []
    actions: list[RecommendedAction] = []
    mechanisms: list[str] = []

    # 1. 轴承温升 / 异常振动相关知识
    bearing_flags = {
        "BEARING_TEMP_DE_TRIP",
        "BEARING_TEMP_DE_WARN",
        "BEARING_TEMP_DE_RAMP_SHARP",
        "BEARING_TEMP_DE_RAMP_SLOW",
        "VIB_DE_TRIP",
        "VIB_DE_WARN",
        "VIB_NDE_TRIP",
        "VIB_NDE_WARN",
        "VIB_BOTH_HIGH",
    }
    if (
        flags.intersection(bearing_flags)
        or "BEARING" in alarm
        or "轴承" in user_query
        or "振动" in user_query
        or "温度" in user_query
    ):
        evidences.append(
            ManualEvidence(
                doc_id="IS100-80-160-IOM",
                chunk_id="chunk_sec4_bearing",
                section="4.2 轴承润滑与温升规范",
                page=28,
                quote="离心泵滚动轴承正常运行温度不得超过65℃，报警温度70℃，跳闸停机极限80℃。温升过快(≥0.8℃/min)通常指示润滑脂干枯、过度充填或滚道严重剥落卡滞。",
                score=0.92,
                channel="HYBRID",
            )
        )
        requirements.append(
            ProcedureRequirement(
                requirement="轴承箱润滑油脂应每运行2000小时补充或更换一次，严禁不同牌号锂基脂混用；每次补充量以充满轴承室1/3~1/2为宜。",
                source_ref="IS100-80-160-IOM#chunk_sec4_bearing",
            )
        )
        actions.append(
            RecommendedAction(
                action="停机冷却后拆检驱动端轴承压盖，检查润滑脂成色（有无炭化、铁屑粉末），测量轴承径向及轴向游隙。",
                priority="HIGH",
                source_ref="IS100-80-160-IOM#chunk_sec4_bearing",
            )
        )
        mechanisms.append("轴承润滑膜破裂导致金属干磨擦，引起瞬态快速温升与高频冲击振动加剧。")

    # 2. 气蚀脱流 / 入口水力失衡相关知识
    cavitation_flags = {"PRESS_IN_TRIP", "PRESS_IN_LOW", "CV_UNSTABLE", "FLOW_DEV"}
    if flags.intersection(cavitation_flags) or "CAVITATION" in alarm or "气蚀" in user_query or "压力" in user_query:
        evidences.append(
            ManualEvidence(
                doc_id="IS100-80-160-IOM",
                chunk_id="chunk_sec3_cavitation",
                section="3.5 泵吸入工况与防气蚀规程",
                page=19,
                quote="装置有效汽蚀余量NPSHa必须大于泵必需汽蚀余量NPSHr+0.5m。入口压力低于-0.04MPa发出预警，低于-0.055MPa将导致流态崩溃发生严重气蚀水击。",
                score=0.88,
                channel="HYBRID",
            )
        )
        requirements.append(
            ProcedureRequirement(
                requirement="定期清洗入口过滤网，严禁通过关小入口管路阀门来调节泵输送流量。",
                source_ref="IS100-80-160-IOM#chunk_sec3_cavitation",
            )
        )
        actions.append(
            RecommendedAction(
                action="排查入口前置过滤器压差，疏通吸入管道堵塞，检查吸入储罐液位以恢复吸入正压。",
                priority="HIGH",
                source_ref="IS100-80-160-IOM#chunk_sec3_cavitation",
            )
        )
        mechanisms.append("入口吸入阻力过大导致介质局部气化，微小气泡进入高压区溃灭产生微射流剥蚀叶轮并伴生剧烈流体脉动。")

    # 3. 密封泄漏相关知识
    if "LEAK" in vision_defect_types or "漏" in user_query:
        evidences.append(
            ManualEvidence(
                doc_id="IS100-80-160-IOM",
                chunk_id="chunk_sec5_seal",
                section="5.1 机械密封检修与泄漏判据",
                page=35,
                quote="机械密封允许微量润滑液外渗（<5滴/min）。若出现连续成线淌漏或喷雾，表明动静环石墨摩擦副碎裂或辅助氟橡胶O形圈高温老化失效。",
                score=0.91,
                channel="HYBRID",
            )
        )
        requirements.append(
            ProcedureRequirement(
                requirement="机械密封安装弹簧压缩量必须精准控制在设计基准±0.5mm，严禁敲击碳化硅/石墨摩擦端面。",
                source_ref="IS100-80-160-IOM#chunk_sec5_seal",
            )
        )
        actions.append(
            RecommendedAction(
                action="切断物料进出口隔离阀，冲洗排放泵腔介质，更换集装式机械密封总成与密封垫片。",
                priority="HIGH",
                source_ref="IS100-80-160-IOM#chunk_sec5_seal",
            )
        )
        mechanisms.append("机械密封动静环干磨擦拉伤或冲洗液中断，导致密封面贴合失效造成介质外泄。")

    # 4. 电机过载 / 欠载
    if flags.intersection({"CURRENT_HIGH", "CURRENT_LOW"}):
        evidences.append(
            ManualEvidence(
                doc_id="IS100-80-160-IOM",
                chunk_id="chunk_sec6_motor",
                section="6.3 电气驱动与负载校核",
                page=42,
                quote="Y160M2-2电机额定电流为22.5A，持续超过29A将导致热继电器保护动作或绝缘烧毁；电流显著偏低通常指示断轴或吸空干转。",
                score=0.86,
                channel="BM25",
            )
        )
        actions.append(
            RecommendedAction(
                action="使用钳形电流表校验三相运行平衡度，核对泵腔内是否存在异物缠绕或机械摩擦阻力过大。",
                priority="MEDIUM",
                source_ref="IS100-80-160-IOM#chunk_sec6_motor",
            )
        )
        mechanisms.append("泵内机械摩擦过载或介质粘度超标造成电机过负荷电流越界。")

    # 兜底常规知识
    if not evidences:
        evidences.append(
            ManualEvidence(
                doc_id="IS100-80-160-IOM",
                chunk_id="chunk_sec2_general",
                section="2.1 离心泵日常点检与巡检基准",
                page=12,
                quote="IS100-80-160离心泵运行期间应按班次巡检听诊轴承运转声响、监测机身振动及轴承温升，记录进出口压力表读数。",
                score=0.75,
                channel="HYBRID",
            )
        )
        actions.append(
            RecommendedAction(
                action="按规定频次落实就地巡检，复测三向振动与轴承表面温度，核实流量计读数。",
                priority="LOW",
                source_ref="IS100-80-160-IOM#chunk_sec2_general",
            )
        )
        mechanisms.append("机泵处于标准运行工况范围，维持计划性预防点检。")

    mechanism_text = "；".join(mechanisms) if mechanisms else "符合常规运行维护规范。"

    manual_state = ManualState(
        status="HIT",
        evidences=evidences,
        mechanism=mechanism_text,
        requirements=requirements,
        actions=actions,
    )
    logger.info("手册知识检索完成: 命中证据=%d, 规程要求=%d, 处置动作=%d", len(evidences), len(requirements), len(actions))
    return {"manual": manual_state}


# =============================================================================
# Reasoner
# =============================================================================


def reasoner_node(state: DiagnosisState) -> dict:
    """
    Step 5：综合证据并推理归因（支持上游预设结果或基于多源证据的启发式归因集成）。
    """
    from src.schemas.state import Hypothesis, ReasoningState

    logger.info("开始综合证据并推理归因")

    # 若调用方或上游已有归因结论且不为初始默认值，优先保留
    if getattr(state.reasoning, "confidence", None) in ("LOW", "MEDIUM", "HIGH") and state.reasoning.root_cause:
        logger.info("保留已有推理结论: 根因=%s, 置信度=%s", state.reasoning.root_cause, state.reasoning.confidence)
        return {}

    flags = set(state.data.threshold_flags or [])
    used_inputs: list[str] = []
    support_sources: list[str] = []
    conflicts: list[str] = []

    # 真实判定使用了哪些输入
    if getattr(state.data.quality, "status", None) in ("OK", "PARTIAL") and (
        state.data.threshold_flags or state.data.metrics or state.data.descriptions
    ):
        used_inputs.append("data")
        support_sources.append("DATA")

    if getattr(state.vision, "status", None) in ("DEFECT_FOUND", "NO_DEFECT") and state.vision.findings:
        used_inputs.append("vision")
        support_sources.append("VISION")

    if getattr(state.manual, "status", None) == "HIT" and state.manual.evidences:
        used_inputs.append("manual")
        support_sources.append("MANUAL")

    if getattr(state.engineer_text, "status", None) == "RECORDED" and state.engineer_text.text.strip():
        used_inputs.append("engineer_text")
        support_sources.append("ENGINEER_TEXT")

    # 归因推导与假设生成
    hypotheses: list[Hypothesis] = []
    root_cause = "机组运行工况综合正常"
    confidence = "LOW"

    # 1. 轴承故障假设
    bearing_crit = {"BEARING_TEMP_DE_TRIP", "BEARING_TEMP_DE_WARN", "BEARING_TEMP_DE_RAMP_SHARP", "VIB_DE_TRIP", "VIB_DE_WARN"}
    if flags.intersection(bearing_crit):
        root_cause = "驱动端轴承干摩擦/疲劳劣化故障"
        confidence = "HIGH" if len(support_sources) >= 2 else "MEDIUM"
        hit_flags = flags.intersection({"BEARING_TEMP_DE_TRIP", "BEARING_TEMP_DE_WARN", "VIB_DE_TRIP"})
        hypotheses.append(
            Hypothesis(
                cause="驱动端轴承润滑失效或疲劳剥落",
                confidence_level="HIGH",
                support_refs=[f"threshold:{f}" for f in hit_flags] or ["threshold:BEARING_TEMP_DE_WARN"],
                status="SUPPORTED",
            )
        )
    # 2. 气蚀脱流假设
    elif flags.intersection({"PRESS_IN_TRIP", "PRESS_IN_LOW", "CV_UNSTABLE"}):
        root_cause = "吸入管路阻力异常导致泵腔气蚀脱流"
        confidence = "HIGH" if len(support_sources) >= 2 else "MEDIUM"
        hit_flags = flags.intersection({"PRESS_IN_TRIP", "PRESS_IN_LOW", "CV_UNSTABLE"})
        hypotheses.append(
            Hypothesis(
                cause="入口吸入绝对压力不足导致介质发生汽蚀",
                confidence_level="HIGH",
                support_refs=[f"threshold:{f}" for f in hit_flags],
                status="SUPPORTED",
            )
        )
    # 3. 密封泄漏假设
    elif any(f.defect_type.upper() == "LEAK" for f in state.vision.findings):
        root_cause = "机械密封动静环摩擦副失效泄漏"
        confidence = "HIGH"
        hypotheses.append(
            Hypothesis(
                cause="机械密封端面贴合失效导致连续淌漏",
                confidence_level="HIGH",
                support_refs=["vision:leak_finding"],
                status="SUPPORTED",
            )
        )
    elif not used_inputs:
        root_cause = "缺乏足够现场数据支持明确归因"
        confidence = "LOW"
    else:
        root_cause = "未见明显恶化异常，各项参数在允许波动范围内"
        confidence = "MEDIUM"

    # 检查跨模态冲突
    if flags.intersection({"BEARING_TEMP_DE_TRIP", "VIB_DE_TRIP"}) and getattr(state.vision, "status", None) == "NO_DEFECT":
        conflicts.append("SCADA 时序触发红线停机阈值，但现场拍摄图片判定无明显机械缺陷。")

    reasoning_state = ReasoningState(
        root_cause=root_cause,
        hypotheses=hypotheses,
        confidence=confidence,
        support_sources=support_sources,
        summary=f"综合【{', '.join(support_sources) if support_sources else '无有效输入'}】得出初步判断：{root_cause}。",
        insufficient_evidence=(len(used_inputs) == 0 or getattr(state.data.quality, "status", None) == "EMPTY"),
        conflicts=conflicts,
        used_inputs=used_inputs,
    )
    logger.info("综合归因完成: 根因=%s, 置信度=%s, 证据源=%s", root_cause, confidence, support_sources)
    return {"reasoning": reasoning_state}


# =============================================================================
# Safety Guard
# =============================================================================


def safety_guard_node(state: DiagnosisState) -> dict:
    """
    Step 6：执行确定性的 RAM 风险和安全门禁卡点与分流。

    这里禁止直接让 LLM 决定：
        safety.decision
        safety.risk_level

    严格使用 rules/safety_guard.py 与 rules/ram_matrix.json 的确定性逻辑产生最终结果。
    """
    from rules.safety_guard import SafetyGuardEngine

    logger.info("开始执行确定性安全门禁评判")
    engine = SafetyGuardEngine.get_instance()

    safety_state = engine.evaluate(state)
    logger.info(
        "安全门禁裁决完成: 决策=%s, 风险等级=%s, 命中规则数=%d",
        safety_state.decision,
        safety_state.risk_level,
        len(safety_state.rule_hits),
    )
    if safety_state.rule_hits:
        logger.warning("安全门禁触发规则代码: %s", safety_state.rule_hits)

    # 依据门禁决策对检修建议实施安全动作防护与覆盖
    protected_actions = engine.apply_action_protection(state, safety_state)
    updated_manual = state.manual.model_copy(update={"actions": protected_actions})

    return {
        "safety": safety_state,
        "manual": updated_manual,
    }


# =============================================================================
# Reporter
# =============================================================================


def reporter_node(state: DiagnosisState) -> dict:
    """
    Step 7：工单交付与分流输出节点。
    根据 safety.decision 实施业务分流：
    - BLOCK: 产生紧急停机指令单，阻断常规自动化派发；
    - INSUFFICIENT_DATA: 产生现场勘验复测清单，阻断正式工单下发；
    - MANUAL_REVIEW: 产生待人工复核工单草稿，锁定自动派发通道；
    - PASS: 产生标准闭环排查工单，允许自动化派单。
    """
    import hashlib
    from src.schemas.state import DeliveryState, EvidenceRef

    decision = getattr(state.safety, "decision", "MANUAL_REVIEW")
    logger.info("开始生成诊断报告，当前安全决策分流=%s", decision)

    diagnosed_at = datetime.now().replace(microsecond=0).isoformat()
    trace_id = state.context.trace_id or "UNKNOWN_TRACE"
    device_id = state.context.device_id or "UNKNOWN_DEVICE"

    # 构建溯源链
    traceability: list[EvidenceRef] = []

    # 1. 时序证据溯源
    for flag in state.data.threshold_flags or []:
        traceability.append(
            EvidenceRef(
                stage="step2_data",
                field="data.threshold_flags",
                source_id=flag,
                locator=f"{state.context.start_time}~{state.context.end_time}",
            )
        )

    # 2. 视觉证据溯源
    for idx, finding in enumerate(state.vision.findings or []):
        traceability.append(
            EvidenceRef(
                stage="step3_vision",
                field="vision.findings",
                source_id=finding.image_id or f"img_{idx}",
                locator=finding.location or "pump_body",
            )
        )

    # 3. 手册证据溯源
    for ev in state.manual.evidences or []:
        traceability.append(
            EvidenceRef(
                stage="step4_manual",
                field="manual.evidences",
                source_id=f"{ev.doc_id}#{ev.chunk_id}",
                locator=f"page_{ev.page}_{ev.section}",
            )
        )

    # 4. 门禁证据溯源
    for rule in state.safety.rule_hits or []:
        traceability.append(
            EvidenceRef(
                stage="step6_safety",
                field="safety.rule_hits",
                source_id=rule,
                locator="rules.safety_guard",
            )
        )

    # 生成确定性内容哈希与交付 URI
    content_raw = f"{trace_id}:{device_id}:{decision}:{state.safety.risk_level}:{len(traceability)}"
    report_hash = hashlib.sha256(content_raw.encode("utf-8")).hexdigest()[:16]

    # 根据分流构建 report_uri
    dispatch_type = {
        "BLOCK": "emergency_shutdown",
        "INSUFFICIENT_DATA": "resurvey_checklist",
        "MANUAL_REVIEW": "manual_review_draft",
        "PASS": "closed_loop_order",
    }.get(decision, "general_order")

    report_uri = f"dispatch://{dispatch_type}/{device_id}/{trace_id}?ref={report_hash}"

    delivery_state = DeliveryState(
        diagnosed_at=diagnosed_at,
        report_ref=report_hash,
        report_uri=report_uri,
        traceability=traceability,
    )
    logger.info("诊断报告交付完成: URI=%s, 证据链条数=%d", report_uri, len(traceability))
    return {"delivery": delivery_state}

