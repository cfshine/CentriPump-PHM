"""确定性安全门禁评判引擎（Safety Guard Engine）。

核心定位与约束：
================
1. 工业级非模型化控制层：严禁调用任何大语言模型（LLM）。
2. 100% 确定性 Python 规则、RAM 风险矩阵与国标/企业技术规范。
3. 拥有无条件拦截、降级、分流上游诊断结论的最高决定权。
4. 完全幂等：相同 DiagnosisState 输入多次计算，保证输出完全一致。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rules.thresholds import (
    FLOW_WARN_LOW_M3H,
    PRESS_IN_TRIP_MPA,
    RULE_CATALOG,
    TEMP_TRIP_C,
    TEMP_WARN_C,
    VIB_TRIP_MM_S,
    VIB_WARN_MM_S,
)
from src.schemas.state import DiagnosisState, RecommendedAction, SafetyState


class SafetyGuardEngine:
    """确定性安全门禁评判引擎。"""

    _instance: SafetyGuardEngine | None = None

    def __init__(self, ram_config_path: Path | str | None = None) -> None:
        if ram_config_path is None:
            ram_config_path = Path(__file__).resolve().parent / "ram_matrix.json"
        self.ram_config_path = Path(ram_config_path)
        self.config = self._load_ram_matrix(self.ram_config_path)

        self.high_risk_devices: set[str] = set(self.config.get("high_risk_devices", []))
        self.trip_rules: set[str] = set(
            self.config.get(
                "trip_rules",
                [
                    "BEARING_TEMP_DE_TRIP",
                    "BEARING_TEMP_DE_RAMP_SHARP",
                    "VIB_DE_TRIP",
                    "VIB_NDE_TRIP",
                    "PRESS_IN_TRIP",
                ],
            )
        )
        self.warn_rules: set[str] = set(
            self.config.get(
                "warn_rules",
                [
                    "BEARING_TEMP_DE_WARN",
                    "BEARING_TEMP_DE_RAMP_SLOW",
                    "VIB_DE_WARN",
                    "VIB_NDE_WARN",
                    "VIB_BOTH_HIGH",
                    "CURRENT_HIGH",
                    "PRESS_IN_LOW",
                    "CV_UNSTABLE",
                ],
            )
        )

    @classmethod
    def get_instance(cls, ram_config_path: Path | str | None = None) -> SafetyGuardEngine:
        """获取引擎单例。"""
        if cls._instance is None:
            cls._instance = cls(ram_config_path)
        return cls._instance

    @staticmethod
    def _load_ram_matrix(path: Path) -> dict[str, Any]:
        """读取 RAM 矩阵及高危设备配置。"""
        if not path.is_file():
            return {
                "high_risk_devices": [
                    "PUMP-A01-HEAVY-OIL",
                    "PUMP-MAIN-CIRC-01",
                    "PUMP-IS100-80-160-CRITICAL",
                ],
                "trip_rules": [
                    "BEARING_TEMP_DE_TRIP",
                    "BEARING_TEMP_DE_RAMP_SHARP",
                    "VIB_DE_TRIP",
                    "VIB_NDE_TRIP",
                    "PRESS_IN_TRIP",
                ],
                "warn_rules": [
                    "BEARING_TEMP_DE_WARN",
                    "BEARING_TEMP_DE_RAMP_SLOW",
                    "VIB_DE_WARN",
                    "VIB_NDE_WARN",
                    "VIB_BOTH_HIGH",
                    "CURRENT_HIGH",
                    "PRESS_IN_LOW",
                    "CV_UNSTABLE",
                ],
            }
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    # =========================================================================
    # 核心评判功能模块（功能 1 ~ 功能 6）
    # =========================================================================

    def _check_critical_trips(self, state: DiagnosisState) -> tuple[bool, list[str], list[str]]:
        """功能 1：极限工况与停机线硬拦截（最高优先级判定）。"""
        tripped = False
        reasons: list[str] = []
        rule_hits: list[str] = []

        # 1. 检查时序规则码中的停机线标志
        flags = set(state.data.threshold_flags or [])
        hit_trip_flags = flags.intersection(self.trip_rules)
        for flag in sorted(hit_trip_flags):
            tripped = True
            rule_hits.append(flag)
            meta = RULE_CATALOG.get(flag, {})
            label = meta.get("label", flag)
            source = meta.get("source", "GB 50275 / 规范第三节")
            reasons.append(f"触发紧急停机线/急剧恶化判据【{flag}】: {label} (依据: {source})。")

        # 2. 检查物理数值直测越界（防范上游规则漏判）
        metrics = state.data.metrics or {}
        overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
        if isinstance(overall, dict):
            max_temp = overall.get("max_temp_de")
            if max_temp is not None and max_temp >= TEMP_TRIP_C:
                if "BEARING_TEMP_DE_TRIP" not in rule_hits:
                    tripped = True
                    rule_hits.append("BEARING_TEMP_DE_TRIP")
                    reasons.append(f"实测驱动端最高温度 {max_temp:.1f}℃ 越过停机线 ({TEMP_TRIP_C}℃)，强制停机拦截。")

            max_vib_de = overall.get("max_vib_de")
            if max_vib_de is not None and max_vib_de >= VIB_TRIP_MM_S:
                if "VIB_DE_TRIP" not in rule_hits:
                    tripped = True
                    rule_hits.append("VIB_DE_TRIP")
                    reasons.append(f"实测驱动端最高振动 {max_vib_de:.2f}mm/s 越过停机线 ({VIB_TRIP_MM_S}mm/s)，强制停机拦截。")

            max_vib_nde = overall.get("max_vib_nde")
            if max_vib_nde is not None and max_vib_nde >= VIB_TRIP_MM_S:
                if "VIB_NDE_TRIP" not in rule_hits:
                    tripped = True
                    rule_hits.append("VIB_NDE_TRIP")
                    reasons.append(f"实测非驱动端最高振动 {max_vib_nde:.2f}mm/s 越过停机线 ({VIB_TRIP_MM_S}mm/s)，强制停机拦截。")

            min_press_in = overall.get("min_press_in")
            if min_press_in is not None and min_press_in <= PRESS_IN_TRIP_MPA:
                if "PRESS_IN_TRIP" not in rule_hits:
                    tripped = True
                    rule_hits.append("PRESS_IN_TRIP")
                    reasons.append(f"实测入口吸上压力 {min_press_in:.3f}MPa 低于脱流停机线 ({PRESS_IN_TRIP_MPA}MPa)，气蚀毁泵先兆。")

        # 3. 检查视觉致命缺陷（例如易燃易爆/强腐蚀介质严重泄漏）
        for finding in state.vision.findings or []:
            if finding.defect_type.upper() == "LEAK" and finding.severity.upper() == "SEVERE":
                tripped = True
                if "RULE_SEVERE_LEAK_TRIP" not in rule_hits:
                    rule_hits.append("RULE_SEVERE_LEAK_TRIP")
                    reasons.append(f"视觉检出严重介质喷淋泄漏 ({finding.location})，触发一级安全阻断。")

        return tripped, reasons, rule_hits

    def _check_data_sufficiency(self, state: DiagnosisState) -> tuple[bool, list[str], list[str]]:
        """功能 2：数据完整性与低置信度硬卡点。"""
        reasons: list[str] = []
        rule_hits: list[str] = []

        # 检查空输入 (完全信息真空)
        data_empty = (
            state.data.quality.status in ("EMPTY", "PENDING")
            and not state.data.threshold_flags
            and not state.data.descriptions
            and not state.data.metrics
        )
        vision_empty = (
            state.vision.status in ("NO_IMAGE", "PENDING")
            and not state.vision.findings
        )
        text_empty = (
            state.engineer_text.status in ("EMPTY", "PENDING")
            or not (state.engineer_text.text.strip() or state.context.user_query.strip())
        )

        if data_empty and vision_empty and text_empty:
            rule_hits.append("RULE_EMPTY_INPUT_GUARD")
            reasons.append("时序数据、现场图片与工程师文字描述全要素缺失，系统处于信息真空状态，拒绝自动派单。")
            return True, reasons, rule_hits

        # 检查虚假输入利用 (大模型幻觉：声称利用了实际不存在的模态)
        used_inputs = [inp.lower() for inp in state.reasoning.used_inputs or []]
        if "data" in used_inputs and state.data.quality.status == "EMPTY":
            rule_hits.append("RULE_INPUT_HALLUCINATION_GUARD")
            reasons.append("Reasoner 声称依赖了时序数据，但实际时序状态为 EMPTY，拦截输入幻觉并强制人工核验。")

        if "vision" in used_inputs and state.vision.status == "NO_IMAGE":
            rule_hits.append("RULE_INPUT_HALLUCINATION_GUARD")
            reasons.append("Reasoner 声称依赖了视觉特征，但实际视觉状态为 NO_IMAGE，拦截输入幻觉并强制人工核验。")

        # 检查数据严重缺损 (PARTIAL 且置信度低)
        if state.data.quality.status == "PARTIAL":
            rule_hits.append("RULE_DATA_PARTIAL_GUARD")
            if state.reasoning.confidence in ("LOW", "PENDING"):
                reasons.append("时序数据采样覆盖率严重不足 (<90%) 且归因置信度偏低，禁止自动放行工单。")

        return False, reasons, rule_hits

    def _is_high_risk_device(self, device_id: str) -> bool:
        """功能 3：高危设备与核心管线黑名单卡点。"""
        if not device_id:
            return False
        clean_id = device_id.strip().upper()
        for pattern in self.high_risk_devices:
            if pattern.upper() in clean_id:
                return True
        return False

    def _check_evidence_conflicts(self, state: DiagnosisState) -> tuple[bool, list[str], list[str]]:
        """功能 4：多源证据逻辑冲突仲裁。"""
        has_conflict = False
        reasons: list[str] = []
        rule_hits: list[str] = []

        # 1. Reasoner 主动上报的证据冲突
        if state.reasoning.conflicts:
            has_conflict = True
            rule_hits.append("RULE_EVIDENCE_CONFLICT")
            for conf in state.reasoning.conflicts:
                reasons.append(f"归因引擎检出跨模态冲突: {conf}")

        # 2. 物理超限 vs 视觉完好冲突（时序严重高温/超振，但视觉声称完好无异常且归因断言正常）
        flags = set(state.data.threshold_flags or [])
        critical_flags = flags.intersection({"BEARING_TEMP_DE_TRIP", "VIB_DE_TRIP"})
        if critical_flags and state.vision.status == "NO_DEFECT":
            if "RULE_EVIDENCE_CONFLICT" not in rule_hits:
                has_conflict = True
                rule_hits.append("RULE_EVIDENCE_CONFLICT")
                reasons.append(
                    f"物理证据严重矛盾: 时序触发红线告警 {list(critical_flags)}，但现场图片判定为无缺陷完好，需人工排查传感器真伪。"
                )

        # 3. 电机过载 vs 水力失压冲突（电气满载 CURRENT_HIGH，但流量极低或入口压力低）
        if "CURRENT_HIGH" in flags:
            metrics = state.data.metrics or {}
            overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
            mean_flow = overall.get("mean_flow")
            if (mean_flow is not None and mean_flow < FLOW_WARN_LOW_M3H) or "PRESS_IN_LOW" in flags:
                if "RULE_EVIDENCE_CONFLICT" not in rule_hits:
                    has_conflict = True
                    rule_hits.append("RULE_EVIDENCE_CONFLICT")
                    reasons.append("机电与水力失衡冲突: 电机电流超额定满载(过负荷)，但出口流量/入口压力严重偏低，疑似断轴、堵塞或空转卡阻。")

        return has_conflict, reasons, rule_hits

    def _evaluate_ram_level(
        self,
        state: DiagnosisState,
        is_high_risk: bool,
        trip_hit: bool,
        has_conflicts: bool,
    ) -> str:
        """功能 5：RAM 风险矩阵评级（S 严重度, O 频次/恶化度, D 可探测度）。"""
        if trip_hit:
            return "CRITICAL"

        flags = set(state.data.threshold_flags or [])
        has_warn_flags = bool(flags.intersection(self.warn_rules))

        # 检查视觉中高度缺陷
        has_moderate_or_severe_vision = any(
            f.severity.upper() in ("SEVERE", "MODERATE") for f in (state.vision.findings or [])
        )

        # 严重度评估 (S)
        if is_high_risk and (has_warn_flags or has_moderate_or_severe_vision):
            return "HIGH"

        if has_warn_flags or has_moderate_or_severe_vision or has_conflicts:
            return "HIGH"

        # 检查轻微缺陷或工况漂移
        has_minor_defect = any(f.severity.upper() == "MINOR" for f in (state.vision.findings or []))
        has_drift = bool(flags.intersection({"FLOW_DEV", "PRESS_OUT_DEV"}))

        if has_minor_defect or has_drift or state.data.quality.status == "PARTIAL":
            return "MEDIUM"

        return "LOW"

    def _derive_decision(
        self,
        risk_level: str,
        is_high_risk: bool,
        has_conflicts: bool,
        has_hallucination: bool,
        data_empty_total: bool,
        state: DiagnosisState,
    ) -> str:
        """功能 6：确定最终安全决策与工作流分流。"""
        # 1. 停机线触发 -> 绝对阻断
        if risk_level == "CRITICAL":
            return "BLOCK"

        # 2. 全要素数据真空 -> 无法派单
        if data_empty_total:
            return "INSUFFICIENT_DATA"

        # 3. 高危设备黑名单 -> 永远不得 PASS，最高只能 MANUAL_REVIEW
        if is_high_risk:
            return "MANUAL_REVIEW"

        # 4. 存在证据冲突或大模型幻觉 -> 强制人工复核
        if has_conflicts or has_hallucination:
            return "MANUAL_REVIEW"

        # 5. 数据有缺损且置信度低 -> 人工介入
        if state.data.quality.status == "PARTIAL" and state.reasoning.confidence in ("LOW", "PENDING"):
            return "MANUAL_REVIEW"

        # 6. 根据 RAM 矩阵等级决策
        if risk_level == "HIGH":
            return "MANUAL_REVIEW"

        if risk_level == "MEDIUM":
            # 只有在归因置信度为 HIGH 且无降级时允许放行，否则转人工
            if state.reasoning.confidence == "HIGH" and not state.workflow.degraded_steps:
                return "PASS"
            return "MANUAL_REVIEW"

        if risk_level == "LOW":
            # 低风险且无红线限制，放行
            return "PASS"

        return "MANUAL_REVIEW"

    # =========================================================================
    # 主评估入口
    # =========================================================================

    def evaluate(self, state: DiagnosisState) -> SafetyState:
        """主入口：执行完整确定性安全门禁评判。"""
        all_reasons: list[str] = []
        all_rule_hits: list[str] = []

        # 1. 检查极限工况停机线（功能 1）
        trip_hit, trip_reasons, trip_rules = self._check_critical_trips(state)
        all_reasons.extend(trip_reasons)
        all_rule_hits.extend(trip_rules)

        # 2. 检查数据完整性与置信度卡点（功能 2）
        data_empty_total, data_reasons, data_rules = self._check_data_sufficiency(state)
        all_reasons.extend(data_reasons)
        all_rule_hits.extend(data_rules)
        has_hallucination = "RULE_INPUT_HALLUCINATION_GUARD" in data_rules

        # 3. 检查高危设备黑名单（功能 3）
        device_id = state.context.device_id or ""
        is_high_risk = self._is_high_risk_device(device_id)
        if is_high_risk:
            all_rule_hits.append("RULE_HIGH_RISK_DEVICE_GUARD")
            all_reasons.append(
                f"设备【{device_id}】属于高危/核心主干管网保护名单，系统剥夺自动放行权限，锁定为人工复核签审。"
            )

        # 4. 检查多源证据冲突（功能 4）
        has_conflicts, conflict_reasons, conflict_rules = self._check_evidence_conflicts(state)
        all_reasons.extend(conflict_reasons)
        all_rule_hits.extend(conflict_rules)

        # 5. RAM 风险矩阵综合评级（功能 5）
        risk_level = self._evaluate_ram_level(
            state=state,
            is_high_risk=is_high_risk,
            trip_hit=trip_hit,
            has_conflicts=has_conflicts,
        )

        # 6. 确定最终分流决策（功能 6）
        decision = self._derive_decision(
            risk_level=risk_level,
            is_high_risk=is_high_risk,
            has_conflicts=has_conflicts,
            has_hallucination=has_hallucination,
            data_empty_total=data_empty_total,
            state=state,
        )

        if not all_reasons:
            all_reasons.append("经 100% 确定性规则及 RAM 矩阵评估，所有关键指标均位于安全绿区，无异常红线触碰。")

        return SafetyState(
            risk_level=risk_level,
            decision=decision,
            rule_hits=sorted(list(set(all_rule_hits))),
            reasons=all_reasons,
        )

    # =========================================================================
    # 防护与分流动作赋能
    # =========================================================================

    def apply_action_protection(
        self, state: DiagnosisState, safety: SafetyState
    ) -> list[RecommendedAction]:
        """依据安全决策对检修建议实施严格的安全防护与动作分流覆盖。

        核心防护机制：
        - 若 BLOCK：无条件压制非紧急建议，强制置顶立即切机停运指令；
        - 若 INSUFFICIENT_DATA：强制覆盖为现场勘验补测清单；
        - 若 MANUAL_REVIEW：强制注入双人审核与数字签名要求；
        - 若 PASS：保留常规闭环排查建议。
        """
        existing_actions = list(state.manual.actions or [])

        if safety.decision == "BLOCK":
            return [
                RecommendedAction(
                    action="【安全门禁最高级指令】立即执行紧急切机停运操作！禁止带病维持运转，防范轴承烧死或介质爆炸事故！",
                    priority="HIGH",
                    source_ref="rules.safety_guard#CRITICAL_BLOCK",
                ),
                RecommendedAction(
                    action="切断机泵主电源并实施安全挂牌上锁 (LOTO)，通知工艺管网值班长启动备用机泵切换。",
                    priority="HIGH",
                    source_ref="rules.safety_guard#LOTO_PROCEDURE",
                ),
            ]

        if safety.decision == "INSUFFICIENT_DATA":
            return [
                RecommendedAction(
                    action="【数据复勘任务】现场补测：使用便携式测振仪采集驱动端/非驱动端三轴向振动加速度及频谱。",
                    priority="HIGH",
                    source_ref="rules.safety_guard#DATA_SUPPLEMENT",
                ),
                RecommendedAction(
                    action="【数据复勘任务】现场补测：使用红外热像仪复测轴承箱表面最高温度，并补充拍摄机械密封动静环高清照片。",
                    priority="HIGH",
                    source_ref="rules.safety_guard#DATA_SUPPLEMENT",
                ),
            ]

        if safety.decision == "MANUAL_REVIEW":
            guard_action = RecommendedAction(
                action="【人工复核卡点】自动下发通道已被安全门禁锁定。须由责任维保工程师完成现场核实并进行数字签字确认。",
                priority="HIGH",
                source_ref="rules.safety_guard#MANUAL_REVIEW_GUARD",
            )
            # 将人工审核置顶，其余建议保留在后供工程师审阅
            return [guard_action] + existing_actions

        # PASS: 放行常规闭环操作
        return existing_actions
