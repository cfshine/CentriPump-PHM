"""State 契约、checkpoint serde 与节点所有权测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.schemas.state import (
    NODE_OUTPUT_WHITELIST,
    AlarmState,
    Confidence,
    DataQuality,
    DataState,
    DeliveryState,
    EvidenceKind,
    EvidenceRef,
    EvidenceSource,
    FailureMode,
    FailureModeProposal,
    Hypothesis,
    Measurement,
    ReasoningState,
    SafetyDecision,
    SafetyVerdict,
    VisionFinding,
    VisionState,
    assert_serde_roundtrip,
    create_initial_state,
    validate_node_update,
)


def _rich_state():
    return create_initial_state(
        trace_id="trace-contract",
        device_id="PUMP-IS100-80-160-01",
        user_query="轴承温度持续升高",
    ).model_copy(
        update={
            "data": DataState(
                quality=DataQuality(status="OK", total_points=120),
                measurements=[
                    Measurement(
                        name="max_temp_de",
                        value=78.5,
                        unit="℃",
                        segment="overall",
                        start_time="2026-09-13T07:30:00",
                        end_time="2026-09-13T08:09:55",
                    )
                ],
                threshold_flags=["BEARING_TEMP_DE_WARN"],
            ),
            "vision": VisionState(
                status="DEFECT_FOUND",
                findings=[
                    VisionFinding(
                        image_id="img-1",
                        defect_type="LEAK",
                        severity="MODERATE",
                        location="机械密封压盖",
                        confidence=Confidence.HIGH,
                        evidence="压盖下方可见连续液流",
                    )
                ],
                summary="发现机械密封渗漏",
            ),
            "reasoning": ReasoningState(
                proposal=FailureModeProposal(
                    failure_mode=FailureMode.BEARING_DEGRADATION,
                    confidence=Confidence.MEDIUM,
                    support_refs=[
                        EvidenceRef(
                            kind=EvidenceKind.THRESHOLD,
                            id="BEARING_TEMP_DE_WARN",
                            locator="rules/thresholds.py",
                        )
                    ],
                ),
                hypotheses=[
                    Hypothesis(
                        failure_mode=FailureMode.BEARING_DEGRADATION,
                        confidence=Confidence.MEDIUM,
                        support_refs=[
                            EvidenceRef(
                                kind=EvidenceKind.THRESHOLD,
                                id="BEARING_TEMP_DE_WARN",
                            )
                        ],
                        status="SUPPORTED",
                    )
                ],
                evidence_sources=[EvidenceSource.DATA, EvidenceSource.VISION],
            ),
            "delivery": DeliveryState(
                diagnosed_at="2026-11-10T12:00:00",
                report_sha256="a" * 64,
                traceability=[
                    EvidenceRef(kind=EvidenceKind.VISION, id="img-1", locator="机械密封压盖")
                ],
            ),
        }
    )


def test_identity_fields_are_required_and_non_empty():
    with pytest.raises(ValidationError):
        create_initial_state(trace_id="", device_id="PUMP-01")
    with pytest.raises(ValidationError):
        create_initial_state(trace_id="trace-1", device_id="")


def test_missing_alarm_codes_are_empty_lists_not_sentinels():
    alarms = AlarmState()
    assert alarms.effective == []
    assert alarms.last == []
    assert alarms.all == []


def test_full_state_survives_checkpoint_serde_roundtrip():
    state = _rich_state()
    assert_serde_roundtrip(state, path="DiagnosisState")


def test_safety_verdict_is_immutable():
    verdict = SafetyVerdict(
        decision=SafetyDecision.BLOCK,
        risk_level="CRITICAL",
        failure_mode=FailureMode.BEARING_DEGRADATION,
        rule_hits=["BEARING_TEMP_DE_TRIP"],
        reasons=["触发停机线"],
        rule_version="1.0.0",
        input_hash="b" * 64,
    )
    with pytest.raises(Exception):
        verdict.decision = SafetyDecision.PASS


def test_node_output_whitelist_accepts_owned_key():
    validate_node_update(
        "reporter",
        {"delivery": DeliveryState(report_sha256="c" * 64)},
    )


def test_node_output_whitelist_rejects_cross_write():
    with pytest.raises(ValueError):
        validate_node_update("data", {"vision": VisionState()})


def test_node_output_requires_registration():
    with pytest.raises(KeyError):
        validate_node_update("unregistered_node", {})


def test_main_graph_nodes_are_all_registered():
    from src.orchestrator.graph import diagnosis_graph

    graph_nodes = set(diagnosis_graph.get_graph().nodes) - {"__start__", "__end__"}
    assert graph_nodes <= set(NODE_OUTPUT_WHITELIST)


def test_nodes_only_return_owned_keys_at_runtime():
    from src.orchestrator.nodes import engineer_text_node, start_node

    state = create_initial_state(
        trace_id="trace-runtime",
        device_id="PUMP-01",
        user_query="现场有异响",
    )
    assert set(engineer_text_node(state)) == {"engineer_text"}
    assert set(start_node(state)) == {"context"}


def test_failure_mode_matrix_keys_match_controlled_vocabulary():
    matrix_path = Path(__file__).resolve().parents[1] / "rules" / "ram_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert set(matrix["failure_mode_matrix"]) == {mode.value for mode in FailureMode}


def test_measurement_requires_unit_and_times_are_optional():
    with pytest.raises(ValidationError):
        Measurement(name="rms", value=12.3, unit="")
    measurement = Measurement(name="rms", value=12.3, unit="mm/s")
    assert measurement.segment == "overall"
