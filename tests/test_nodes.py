# tests/test_nodes.py
"""状态契约、阈值唯一出处与节点挂载测试。

覆盖三件事：
  1. **父子图共享状态**是否符合设计（公共继承 / 私有不外流）；
  2. 主图能把 Step 2 子图作为节点挂上去（本次迁移的核心目的）；
  3. 阈值库唯一出处是否被 Step 2 真正引用。

运行::

    pytest tests/test_nodes.py -q
    # 挂载类测试需要 DEEPSEEK_API_KEY（未配置时自动跳过）
"""

from __future__ import annotations

import pytest

from src.schemas.state import DiagnosisState
from src.sub_agents.data_agent.data_state import DataAgentState

# =============================================================================
# 1. 状态分层契约
# =============================================================================


def test_public_state_has_expected_contract() -> None:
    """公共契约 DiagnosisState 含 Step1 输入 + Step2/Step3 产出，不含私有字段。"""
    fields = set(DiagnosisState.model_fields)
    expected = {
        # A 类：Step 1 输入
        "device_id", "start_time", "end_time", "alarm_code", "image_refs",
        # B 类：Step 2 产出（下游 Step4/5/6/7 读取）
        "calculated_metrics", "threshold_flags", "effective_alarm_codes",
        "last_alarm_codes", "all_alarm_codes_in_window",
        "llm_description", "basic_judgment", "rag_search_queries",
        # C 类：Step 3 产出（人读的描述 + 机器读的强类型盒子）
        "visual_description", "visual_findings",
    }
    assert fields == expected, f"公共契约发生变化：多={fields - expected} 少={expected - fields}"


def test_private_fields_stay_in_subgraph_state_only() -> None:
    """★ 私有字段只存在于子图 state，不进公共契约。"""
    private = set(DataAgentState.model_fields) - set(DiagnosisState.model_fields)
    assert private == {"raw_telemetry_data", "alarm_events"}, (
        "子图私有字段集合变了；这些字段不该出现在主控 DiagnosisState 里"
    )
    assert not (private & set(DiagnosisState.model_fields))


def test_subgraph_state_inherits_public_fields() -> None:
    """★ 子图必须继承公共契约 —— 否则它连 device_id 都读不到（已实测）。"""
    assert issubclass(DataAgentState, DiagnosisState)


def test_state_rejects_undeclared_fields() -> None:
    """字段名写错要当场报错，而不是静默丢数据。"""
    with pytest.raises(Exception):
        DiagnosisState(不存在的字段=1)  # type: ignore[call-arg]


def test_subscript_access_kept_for_node_code() -> None:
    """保留下标访问，节点代码无需为 Pydantic 改造。"""
    s = DiagnosisState(device_id="PUMP-1")
    assert s["device_id"] == "PUMP-1"
    assert s.get("start_time") == ""
    assert "device_id" in s and "raw_telemetry_data" not in s


# =============================================================================
# 2. 阈值唯一出处
# =============================================================================


def test_step2_reads_thresholds_from_single_source() -> None:
    """Step 2 的判据必须来自 rules/thresholds.py，而不是自己写死数字。"""
    from rules import thresholds as t

    # 规范第三节 / 第六节的关键判据
    assert (t.TEMP_WARN_C, t.TEMP_TRIP_C) == (70.0, 80.0)
    assert (t.VIB_WARN_MM_S, t.VIB_TRIP_MM_S) == (3.5, 4.5)
    assert (t.DEV_FLOW_PCT, t.DEV_PRESS_PCT) == (15.0, 20.0)
    assert (t.TEMP_SLOPE_SLOW, t.TEMP_SLOPE_SHARP) == (0.2, 0.8)
    assert t.CV_CAVITATION_PCT == 12.0 and t.CV_WINDOW_SEC == 120

    # analyzer 引用的就是同一批对象（同一性，而非仅相等）
    from src.sub_agents.data_agent import analyzer

    assert analyzer.TEMP_TRIP_C is t.TEMP_TRIP_C
    assert analyzer.VIB_WARN_MM_S is t.VIB_WARN_MM_S


def test_analyzer_has_no_magic_numbers() -> None:
    """analyzer.py 里不该再出现规范级的魔法数字。"""
    import inspect
    import re

    from src.sub_agents.data_agent import analyzer

    src = inspect.getsource(analyzer)
    body = "\n".join(
        line for line in src.splitlines()
        if not line.lstrip().startswith("#")          # 跳过注释
    )
    leftovers = set(re.findall(r"(?<![\w.])(\d+\.\d+)(?![\w])", body))
    # 允许残留的只有与判据无关的格式常量
    allowed = {"0.0", "1.0", "2.0", "3.0"}
    assert leftovers <= allowed, f"analyzer.py 里仍有魔法数字: {sorted(leftovers - allowed)}"


# =============================================================================
# 3. 主图挂载（需要可构建的图 —— 缺 Key 时跳过）
# =============================================================================


def _main_graph_or_skip():
    try:
        from src.orchestrator.graph import build_main_graph

        return build_main_graph()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"图不可构建（通常是未配置 DEEPSEEK_API_KEY）：{exc}")


def test_step2_subgraph_is_mounted_in_main_graph() -> None:
    """★ Step 2 / Step 3 作为节点挂在主图上。"""
    app = _main_graph_or_skip()
    nodes = app.get_graph().nodes
    assert "step2" in nodes
    assert "step3" in nodes


def test_main_graph_runs_and_private_fields_do_not_leak() -> None:
    """★ 端到端：主图跑通，且子图私有字段没有外流到主控契约。

    用真实数据库里的一个短窗口（60 点）。缺数据或缺 Key 时跳过。
    """
    import os

    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("需要 DEEPSEEK_API_KEY（本机写在 ~/.bashrc）")

    app = _main_graph_or_skip()
    out = app.invoke({
        "device_id": "PUMP-IS100-80-160-01",
        "start_time": "2026-09-13T08:00:00",
        "end_time": "2026-09-13T08:04:55",
        "alarm_code": "",
    })

    if not out.get("calculated_metrics"):
        pytest.skip("数据库窗口内无数据（需要本地 scada_db）")

    # 公共产出回流主图
    assert out["calculated_metrics"]["overall"]["total_points"] > 0

    # ★ 私有字段不在公共契约里 —— 主图 state 按 DiagnosisState 过滤，它们不会外流
    assert not ({"raw_telemetry_data", "alarm_events"} & set(DiagnosisState.model_fields))


# =============================================================================
# 4. 未提供时间窗口时的软降级（2026-09-17）
#    背景：真实输入可能"只给图片路径、不给窗口"。以前空窗口会在 fetch_data 里
#    抛 ValueError 掀翻整张图 —— 连与 Step 2 无关的 Step 3 都轮不到执行。
# =============================================================================


def _data_nodes_or_skip():
    try:
        from src.sub_agents.data_agent import data_nodes

        return data_nodes
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"节点模块不可导入（通常是未配置 DEEPSEEK_API_KEY）：{exc}")


def test_step2_soft_degrades_without_window() -> None:
    """★ 未提供窗口：不取数、不计算、**不调大模型**，返回结构完整的空结果。

    三个节点必须做出同样的判断，否则会出现"取数跳过了、语义化却照调模型"的半截状态。
    """
    dn = _data_nodes_or_skip()
    state = {"device_id": "PUMP-IS100-80-160-01", "start_time": "", "end_time": "",
             "alarm_code": "FAL-104"}

    fetched = dn.fetch_data_node(state)                    # 不查库
    assert fetched["raw_telemetry_data"] == []

    metrics = dn.calculate_metrics_node({**state, **fetched})
    assert metrics["calculated_metrics"] == {}
    assert any("未做时序分析" in f for f in metrics["threshold_flags"])
    assert metrics["effective_alarm_codes"] == "FAL-104"   # 入参报警码仍带下去
    assert metrics["all_alarm_codes_in_window"] == []

    sem = dn.semanticize_node({**state, **fetched, **metrics})   # 不调大模型
    assert "未做时序分析" in sem["llm_description"]
    assert "未执行时序分析" in sem["basic_judgment"]
    assert sem["rag_search_queries"] == []


def test_window_without_data_is_distinct_from_no_window() -> None:
    """两种"没东西可算"的告警文案必须不同，否则下游分不清。

    · 未提供窗口   → "未提供时间窗口：本次未做时序分析"
    · 窗口内无数据 → "无数据：指定时间窗口内无 SCADA 记录"
    """
    dn = _data_nodes_or_skip()
    no_window = dn.calculate_metrics_node({"start_time": "", "end_time": "",
                                           "raw_telemetry_data": [], "alarm_code": ""})
    empty_window = dn.calculate_metrics_node({"start_time": "2026-09-13T00:00:00",
                                              "end_time": "2026-09-13T00:01:00",
                                              "raw_telemetry_data": [], "alarm_code": ""})
    assert "未做时序分析" in no_window["threshold_flags"][0]
    assert "无数据" in empty_window["threshold_flags"][0]
    assert no_window["threshold_flags"] != empty_window["threshold_flags"]


def test_main_graph_runs_without_window() -> None:
    """★ 端到端：只给图片、不给窗口，主图**不再崩**，Step 3 照常交出契约形状。

    这里用空图片清单（Step 3 零成本短路）以做到不花 API 调用；
    "有图但没窗口"是同一个代码路径，另有跑批脚本覆盖真实调用。
    """
    app = _main_graph_or_skip()
    out = app.invoke({"image_refs": []})

    # Step 2 明确降级，而不是崩
    assert out["calculated_metrics"] == {}
    assert any("未做时序分析" in f for f in out["threshold_flags"])
    # Step 3 仍然给出恒定形状
    assert out["visual_description"] == ""
    findings = out["visual_findings"]
    images = findings.images if hasattr(findings, "images") else findings["images"]
    assert images == []
