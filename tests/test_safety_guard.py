"""Step 6 确定性安全门禁（Safety Guard）与分流交付端到端测试。

覆盖：
1. 红线拦截；
2. 高危设备黑名单；
3. 空输入拒止；
4. 证据冲突仲裁；
5. 输入幻觉拦截；
6. 正常放行；
7. 完全幂等 + 输入哈希可重放；
8. LangGraph 端到端分流。
"""

from uuid import uuid4

import pytest

from rules.safety_guard import SafetyGuardEngine
from src.schemas.state import (
    Confidence,
    DataQuality,
    DataState,
    DiagnosisState,
    EngineerTextState,
    EvidenceSource,
    FailureMode,
    FailureModeProposal,
    Measurement,
    ReasoningState,
    VisionState,
    create_initial_state,
)


@pytest.fixture
def engine() -> SafetyGuardEngine:
    return SafetyGuardEngine.get_instance()


def _m(name: str, value: float, unit: str) -> Measurement:
    return Measurement(name=name, value=value, unit=unit, segment="overall")


def _data(*, status: str = "OK", flags: list[str] | None = None, measurements=None, reason=""):
    return DataState(
        quality=DataQuality(status=status, total_points=120 if status == "OK" else 0, reason=reason),
        threshold_flags=list(flags or []),
        measurements=list(measurements or []),
    )


def _reasoning(
    mode: FailureMode,
    confidence: Confidence,
    sources: list[EvidenceSource] | None = None,
    *,
    conflicts: list[str] | None = None,
    insufficient: bool = False,
) -> ReasoningState:
    return ReasoningState(
        proposal=FailureModeProposal(
            failure_mode=mode,
            confidence=confidence,
        ),
        evidence_sources=list(sources or []),
        conflicts=list(conflicts or []),
        insufficient_evidence=insufficient,
    )


def test_critical_trip_interception(engine: SafetyGuardEngine):
    """停机线告警必须无条件 BLOCK，并压制上游乐观提议。"""
    state = create_initial_state(
        trace_id=f"test-{uuid4().hex[:8]}",
        device_id="PUMP-IS100-80-160-01",
    ).model_copy(
        update={
            "data": _data(
                flags=["BEARING_TEMP_DE_TRIP", "VIB_DE_TRIP"],
                measurements=[_m("max_temp_de", 84.5, "℃"), _m("max_vib_de", 4.8, "mm/s")],
            ),
            "reasoning": _reasoning(
                FailureMode.NORMAL,
                Confidence.HIGH,
                [EvidenceSource.DATA],
            ),
        }
    )

    safety = engine.evaluate(state)
    assert safety.decision == "BLOCK"
    assert safety.risk_level == "CRITICAL"
    assert safety.failure_mode == FailureMode.BEARING_DEGRADATION
    assert "BEARING_TEMP_DE_TRIP" in safety.rule_hits
    assert "VIB_DE_TRIP" in safety.rule_hits
    assert any("停机" in r for r in safety.reasons)

    protected_actions = engine.apply_action_protection(state, safety)
    assert len(protected_actions) >= 1
    assert "紧急切机停运" in protected_actions[0].action
    assert protected_actions[0].priority == "HIGH"


def test_high_risk_device_blacklist(engine: SafetyGuardEngine):
    """高危设备永远不得 PASS，最高 MANUAL_REVIEW。"""
    state = create_initial_state(
        trace_id=f"test-{uuid4().hex[:8]}",
        device_id="PUMP-A01-HEAVY-OIL",
    ).model_copy(
        update={
            "data": _data(measurements=[_m("max_temp_de", 45.0, "℃")]),
            "vision": VisionState(status="NO_DEFECT", findings=[]),
            "reasoning": _reasoning(
                FailureMode.NORMAL,
                Confidence.HIGH,
                [EvidenceSource.DATA, EvidenceSource.VISION],
            ),
        }
    )

    safety = engine.evaluate(state)
    assert safety.decision == "MANUAL_REVIEW"
    assert "RULE_HIGH_RISK_DEVICE_GUARD" in safety.rule_hits
    assert any("高危" in r for r in safety.reasons)

    protected_actions = engine.apply_action_protection(state, safety)
    assert any("人工复核卡点" in a.action for a in protected_actions)


def test_empty_input_insufficient_data(engine: SafetyGuardEngine):
    """时序、图片、工程师描述全空时必须 INSUFFICIENT_DATA。"""
    state = create_initial_state(
        trace_id=f"test-{uuid4().hex[:8]}",
        device_id="PUMP-IS100-80-160-01",
    )

    assert state.data.quality.status == "PENDING"
    assert state.vision.status == "PENDING"
    assert state.engineer_text.status == "PENDING"

    safety = engine.evaluate(state)
    assert safety.decision == "INSUFFICIENT_DATA"
    assert safety.failure_mode == FailureMode.INSUFFICIENT_EVIDENCE
    assert "RULE_EMPTY_INPUT_GUARD" in safety.rule_hits
    assert any("信息真空" in r for r in safety.reasons)

    protected_actions = engine.apply_action_protection(state, safety)
    assert any("数据复勘任务" in a.action for a in protected_actions)


def test_evidence_conflict_arbitration(engine: SafetyGuardEngine):
    """时序超限但视觉完好时，必须命中证据冲突规则。"""
    state = create_initial_state(
        trace_id=f"test-{uuid4().hex[:8]}",
        device_id="PUMP-IS100-80-160-01",
    ).model_copy(
        update={
            "data": _data(
                flags=["BEARING_TEMP_DE_TRIP"],
                measurements=[_m("max_temp_de", 82.0, "℃")],
            ),
            "vision": VisionState(status="NO_DEFECT", findings=[]),
            "reasoning": _reasoning(
                FailureMode.BEARING_DEGRADATION,
                Confidence.MEDIUM,
                [EvidenceSource.DATA, EvidenceSource.VISION],
                conflicts=["SCADA显示轴承82℃，但现场近景红外与漆面无变色痕迹"],
            ),
        }
    )

    safety = engine.evaluate(state)
    assert "RULE_EVIDENCE_CONFLICT" in safety.rule_hits
    assert safety.decision in ("BLOCK", "MANUAL_REVIEW")
    assert any("矛盾" in r or "冲突" in r for r in safety.reasons)


def test_input_hallucination_detection(engine: SafetyGuardEngine):
    """Reasoner 声称使用实际不存在的模态时，必须命中输入幻觉规则。"""
    state = create_initial_state(
        trace_id=f"test-{uuid4().hex[:8]}",
        device_id="PUMP-IS100-80-160-01",
    ).model_copy(
        update={
            "data": _data(status="EMPTY", reason="无数据"),
            "engineer_text": EngineerTextState(status="RECORDED", text="巡检发现声音异常"),
            "reasoning": _reasoning(
                FailureMode.BEARING_DEGRADATION,
                Confidence.HIGH,
                [EvidenceSource.DATA, EvidenceSource.ENGINEER_TEXT],
            ),
        }
    )

    safety = engine.evaluate(state)
    assert "RULE_INPUT_HALLUCINATION_GUARD" in safety.rule_hits
    assert safety.decision == "MANUAL_REVIEW"
    assert any("幻觉" in r for r in safety.reasons)


def test_normal_pass_decision(engine: SafetyGuardEngine):
    """指标正常、证据充分、置信度高时自动放行。"""
    state = create_initial_state(
        trace_id=f"test-{uuid4().hex[:8]}",
        device_id="PUMP-IS100-80-160-NORMAL",
    ).model_copy(
        update={
            "data": _data(
                measurements=[_m("max_temp_de", 45.0, "℃"), _m("max_vib_de", 1.1, "mm/s")]
            ),
            "vision": VisionState(status="NO_DEFECT", findings=[]),
            "reasoning": _reasoning(
                FailureMode.NORMAL,
                Confidence.HIGH,
                [EvidenceSource.DATA, EvidenceSource.VISION],
            ),
        }
    )

    safety = engine.evaluate(state)
    assert safety.decision == "PASS"
    assert safety.risk_level == "LOW"
    assert safety.failure_mode == FailureMode.NORMAL


def test_safety_guard_idempotency_and_replay_hash(engine: SafetyGuardEngine):
    """相同输入重复评估必须完全一致，且 input_hash 可复现。"""
    state = create_initial_state(
        trace_id=f"test-{uuid4().hex[:8]}",
        device_id="PUMP-IS100-80-160-01",
    ).model_copy(
        update={
            "data": _data(
                flags=["BEARING_TEMP_DE_WARN", "VIB_DE_WARN"],
                measurements=[_m("max_temp_de", 72.0, "℃"), _m("max_vib_de", 3.8, "mm/s")],
            ),
            "reasoning": _reasoning(
                FailureMode.BEARING_DEGRADATION,
                Confidence.MEDIUM,
                [EvidenceSource.DATA],
            ),
        }
    )

    first_eval = engine.evaluate(state).model_dump()
    for _ in range(100):
        assert engine.evaluate(state).model_dump() == first_eval

    verdict = engine.evaluate(state).verdict
    assert verdict is not None
    assert verdict.rule_version == engine.rule_version
    assert verdict.input_hash == engine.evaluate(state).input_hash


def test_graph_end_to_end_shunting():
    """LangGraph 完整主图端到端分流。"""
    from src.orchestrator.graph import diagnosis_graph

    # 停机线拦截场景
    trace_id_block = f"e2e-block-{uuid4().hex[:8]}"
    state_block = create_initial_state(
        trace_id=trace_id_block,
        device_id="PUMP-IS100-80-160-01",
        user_query="泵体有剧烈啸叫且表面烫手",
    ).model_copy(
        update={
            "data": _data(
                flags=["BEARING_TEMP_DE_TRIP", "VIB_DE_TRIP"],
                measurements=[_m("max_temp_de", 85.0, "℃"), _m("max_vib_de", 5.2, "mm/s")],
            )
        }
    )

    result_block = diagnosis_graph.invoke(
        state_block,
        config={"configurable": {"thread_id": trace_id_block}},
    )
    final_block: DiagnosisState = DiagnosisState.model_validate(result_block)

    assert final_block.safety.decision == "BLOCK"
    assert final_block.safety.risk_level == "CRITICAL"
    assert "emergency_shutdown" in final_block.delivery.report_uri
    assert final_block.delivery.report_sha256
    assert any("紧急切机" in a.action for a in final_block.manual.actions)
    assert len(final_block.delivery.traceability) > 0

    # 空数据拒止场景
    trace_id_empty = f"e2e-empty-{uuid4().hex[:8]}"
    state_empty = create_initial_state(
        trace_id=trace_id_empty,
        device_id="PUMP-IS100-80-160-01",
    ).model_copy(
        update={
            "data": _data(status="EMPTY", reason="窗口内无数据"),
            "vision": VisionState(status="NO_IMAGE", findings=[]),
            "engineer_text": EngineerTextState(status="EMPTY"),
        }
    )
    result_empty = diagnosis_graph.invoke(
        state_empty,
        config={"configurable": {"thread_id": trace_id_empty}},
    )
    final_empty: DiagnosisState = DiagnosisState.model_validate(result_empty)

    assert final_empty.safety.decision == "INSUFFICIENT_DATA"
    assert "resurvey_checklist" in final_empty.delivery.report_uri
    assert any("数据复勘任务" in a.action for a in final_empty.manual.actions)
