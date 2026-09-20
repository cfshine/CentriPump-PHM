# tests/test_nodes.py
"""状态契约、阈值唯一出处与节点行为测试（适配组长的 9 盒子契约）。

覆盖五件事：
  1. **公共契约**：DiagnosisState 是 9 个盒子；Step 2 只写 ``data`` 一个盒子；
  2. **子图结构**：2 个节点（analyze / summarize），不需要私有 state；
  3. **阈值唯一出处**：判据与规则码表都来自 ``rules/thresholds.py``；
  4. **节点行为**：软降级、报警码来源、规则码、以及"回写不能冲掉同盒子其他字段"；
  5. **挂载**：子图能挂进一个迷你父图，且**回流的键不超出契约**。

运行::

    pytest tests/test_nodes.py -q
    # 涉及建图/取数的用例需要 DEEPSEEK_API_KEY（未配置时自动跳过）

★ 取数类用例用 monkeypatch 注入合成帧，不需要 MySQL ——
  "软降级 / 窗口无数据 / 报警码只认数据 / 规则码" 全部可离线验证。
"""

from __future__ import annotations

import importlib.util

import pytest

from src.schemas.state import DataDescription, DataState, DiagnosisState, create_initial_state

#: 组长的 9 个一级盒子（公共契约的顶层结构）
TOP_LEVEL_BOXES = {
    "context", "data", "vision", "manual", "reasoning",
    "safety", "human", "delivery", "workflow",
}


def _state(device_id: str = "PUMP-IS100-80-160-01", start: str = "",
           end: str = "", alarm_code: str = "") -> DiagnosisState:
    """按组长的构造函数造一个真实的状态对象（不能再用普通 dict：节点读的是属性）。"""
    return create_initial_state(
        trace_id="test-trace",
        device_id=device_id,
        start_time=start,
        end_time=end,
        alarm_code=alarm_code,
    )


# =============================================================================
# 1. 公共契约（组长的 9 盒子）
# =============================================================================


def test_state_is_nine_boxes() -> None:
    """顶层就是那 9 个盒子，Step 2 的产出不再是顶层扁平字段。"""
    assert set(DiagnosisState.model_fields) == TOP_LEVEL_BOXES


def test_step2_writes_only_the_data_box() -> None:
    """Step 2 的家在 ``data``：6 个子字段的集合不许漂移。"""
    assert set(DataState.model_fields) == {
        "telemetry_ref", "quality", "metrics", "threshold_flags", "alarms", "descriptions",
    }


def test_flat_step2_fields_are_gone() -> None:
    """★ 旧的扁平字段不许回到顶层（它们已经归到 data 盒子里）。"""
    flat = {
        "calculated_metrics", "threshold_flags", "llm_description",
        "effective_alarm_codes", "last_alarm_codes", "all_alarm_codes_in_window",
        "basic_judgment", "rag_search_queries", "visual_description", "visual_findings",
    }
    still_there = flat & set(DiagnosisState.model_fields)
    assert not still_there, f"这些扁平字段不该出现在顶层：{sorted(still_there)}"


def test_state_rejects_undeclared_fields() -> None:
    """字段名写错要当场报错，而不是静默丢数据（extra='forbid'）。"""
    with pytest.raises(Exception):
        DiagnosisState(不存在的字段=1)  # type: ignore[call-arg]


def test_state_has_no_subscript_access() -> None:
    """★ 组长的 StateModel **没有** ``__getitem__`` / ``get()``。

    这条是防回归：节点代码必须写 ``state.context.xxx``，
    写成 ``state["start_time"]`` 或 ``state.get("start_time")`` 会在运行时炸。
    """
    s = _state()
    with pytest.raises(TypeError):
        _ = s["context"]  # type: ignore[index]
    assert not hasattr(s, "get")


# =============================================================================
# 2. 子图结构：2 个节点、无私有 state
# =============================================================================


def _data_graph_or_skip():
    try:
        from src.sub_agents.data_agent import data_graph

        return data_graph
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"子图不可导入（通常是未配置 DEEPSEEK_API_KEY）：{exc}")


def _data_nodes_or_skip():
    try:
        from src.sub_agents.data_agent import data_nodes

        return data_nodes
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"节点模块不可导入（通常是未配置 DEEPSEEK_API_KEY）：{exc}")


def test_subgraph_has_two_nodes() -> None:
    """★ 取数与确定性计算已合并 → analyze / summarize 两个节点。"""
    dg = _data_graph_or_skip()
    nodes = set(dg.data_agent_graph.get_graph().nodes)
    assert {"analyze", "summarize"} <= nodes
    assert not ({"fetch_data", "calculate_metrics", "semanticize"} & nodes), (
        f"旧的 3 节点名字还在：{sorted(nodes)}"
    )


def test_private_state_module_is_gone() -> None:
    """★ Step 2 已无私有字段 → DataAgentState 那一层被删掉。

    子图直接拿公共契约 ``DiagnosisState`` 当自己的 state；
    将来哪个子 Agent 真的需要私有字段，再按"继承公共契约 + 补私有字段"扩展。
    """
    assert importlib.util.find_spec("src.sub_agents.data_agent.data_state") is None, (
        "data_state.py 应已删除（Step 2 没有私有字段了）"
    )


# =============================================================================
# 3. 阈值唯一出处 + 规则码表
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


def test_analyzer_flags_are_facts_not_verdicts() -> None:
    """★ 规则命中只输出机器码，不输出分级/归因措辞。

    2026-09-17 去掉的两类措辞（现在连中文句都没有了，由渲染器在 prompt 侧生成）：
      · "红色紧急 / 黄色关注"  —— 严重度分级属于处置侧（Step 6）的判断；
      · "疑似流态失稳"        —— 归因性结论，Step 2 只描述状态。
    """
    import inspect

    from src.sub_agents.data_agent import analyzer

    src = inspect.getsource(analyzer)
    for banned in ("红色紧急", "黄色关注", "疑似"):
        assert banned not in src, f"analyzer.py 里不该再出现分级/归因措辞：{banned}"


def test_every_emittable_code_exists_in_rule_catalog() -> None:
    """★ 防漂移：analyzer 里 ``hits.append("XXX")`` 的码必须与 RULE_CATALOG 一一对应。

    这条同时挡住两种情况：① 代码里新增了码但忘了登记；② 码表里留了没人用的孤儿。
    """
    import inspect
    import re

    from rules.thresholds import RULE_CATALOG
    from src.sub_agents.data_agent import analyzer

    emitted = set(re.findall(r'hits\.append\("([A-Z_]+)"\)', inspect.getsource(analyzer)))
    assert emitted, "没扫到任何规则码，检查 analyzer 的写法是否变了"
    assert emitted <= set(RULE_CATALOG), f"码表里缺：{sorted(emitted - set(RULE_CATALOG))}"
    assert set(RULE_CATALOG) == emitted, f"码表里有多余项：{sorted(set(RULE_CATALOG) - emitted)}"


def test_render_rule_hits_turns_codes_into_chinese_facts() -> None:
    """渲染器：把机器码渲染成给大模型看的中文事实句（只进 prompt，不进 state）。"""
    from src.sub_agents.data_agent.timeseries_tools import render_rule_hits

    phases = [
        {"state": "WARNING", "start": "2026-09-13T08:00:05", "end": "2026-09-13T08:09:55",
         "ramp_start_time": "2026-09-13T08:01:00",
         "rule_hits": ["BEARING_TEMP_DE_TRIP", "BEARING_TEMP_DE_RAMP_SHARP"]},
        {"state": "NORMAL", "start": "2026-09-13T08:10:00", "end": "2026-09-13T08:10:05",
         "rule_hits": []},
    ]
    text = render_rule_hits(phases)

    assert "驱动端温度超停机线（BEARING_TEMP_DE_TRIP）" in text     # 码 + 中文标签
    assert "约从 08:01:00 开始" in text                              # extra_field 生效
    assert "[WARNING段 08:00:05~08:09:55]" in text                   # 段定位
    assert "[NORMAL段" not in text                                   # 没命中的段不出现
    assert render_rule_hits([]) == "无"


# =============================================================================
# 4. 报警事件查询已删除 —— 每个窗口只查一次库
# =============================================================================


def test_alarm_event_query_is_gone() -> None:
    """★ 报警事件查询（含游程编码/去抖/段数上限）整块删除。"""
    from src.sub_agents.data_agent import data_tools

    assert not hasattr(data_tools, "query_alarm_events")
    assert not hasattr(data_tools, "_summarize_run")
    assert not hasattr(data_tools, "_cap_runs")


# =============================================================================
# 5. 节点行为（用 monkeypatch 注入合成帧，不需要 MySQL）
# =============================================================================


def _row(ts: str, state: str = "NORMAL", alarm_code: str = "NONE", **over):
    """造一帧合成遥测（字段与 query_scada_telemetry 的返回一致）。"""
    row = {
        "timestamp": ts,
        "flow_rate": 100.0,
        "press_out": 0.312,
        "press_in": 0.020,
        "temp_de": 45.0,
        "temp_nde": 43.0,
        "vib_rms_de": 1.45,
        "vib_rms_nde": 1.35,
        "motor_current": 22.5,
        "operating_state": state,
        "alarm_code": alarm_code,
    }
    row.update(over)
    return row


def test_step2_soft_degrades_without_window(monkeypatch) -> None:
    """★ 未提供窗口：不取数、不计算、**不调大模型**，降级信息落在 quality.reason。"""
    dn = _data_nodes_or_skip()

    def _boom(*_a, **_k):  # 未提供窗口时**不该**有任何查询
        raise AssertionError("未提供时间窗口时不该查库")

    monkeypatch.setattr(dn, "query_scada_telemetry", _boom)

    s = _state(alarm_code="FAL-104")
    out = dn.analyze_node(s)
    assert list(out) == ["data"]                                   # 只回写 data 一个盒子
    assert out["data"].quality.status == "EMPTY"
    assert "未做时序分析" in out["data"].quality.reason
    assert out["data"].metrics == {}

    # 模拟 LangGraph 把 analyze 的回写合并进 state，再跑 summarize
    s2 = s.model_copy(update=out)
    assert dn.summarize_node(s2) == {}                             # 不写、也不调大模型


def test_window_without_data_is_distinct_from_no_window(monkeypatch) -> None:
    """两种"没东西可算"必须能区分，否则下游分不清"没做"和"做了但没数据"。"""
    dn = _data_nodes_or_skip()
    monkeypatch.setattr(dn, "query_scada_telemetry", lambda *a, **k: [])

    no_window = dn.analyze_node(_state())
    empty_window = dn.analyze_node(_state(start="2026-09-13T00:00:00",
                                          end="2026-09-13T00:01:00"))

    assert no_window["data"].quality.status == "EMPTY"
    assert empty_window["data"].quality.status == "EMPTY"
    assert "未提供时间窗口" in no_window["data"].quality.reason
    assert "无 SCADA 记录" in empty_window["data"].quality.reason
    assert no_window["data"].quality.reason != empty_window["data"].quality.reason


def test_data_update_only_writes_data_box_and_keeps_other_fields(monkeypatch) -> None:
    """★ 回写不能冲掉同盒子里的其他字段，也不能多写别的盒子。

    这是实测过的坑：LangGraph 会用返回值**替换整个盒子**，
    如果 summarize 只返回 ``{"data": {"descriptions": ...}}``，
    analyze 写好的 quality / metrics / alarms 就全没了。
    """
    dn = _data_nodes_or_skip()
    monkeypatch.setattr(dn, "query_scada_telemetry", lambda *a, **k: [
        _row("2026-09-13T00:00:00"), _row("2026-09-13T00:00:05"),
    ])

    s = _state(start="2026-09-13T00:00:00", end="2026-09-13T00:00:05")
    first = dn.analyze_node(s)                      # 先写 quality / metrics / alarms …
    s2 = s.model_copy(update=first)
    merged = dn._data_update(s2, descriptions=[
        DataDescription(type="OTHER", description="占位", evidence=[])
    ])

    assert list(merged) == ["data"]                             # 只回写 data
    d = merged["data"]
    assert d.quality.status == "OK"                             # 没被冲掉
    assert d.metrics["overall"]["total_points"] == 2            # 没被冲掉
    assert d.threshold_flags == first["data"].threshold_flags   # 没被冲掉
    assert d.descriptions[0].description == "占位"               # 新字段生效


def test_data_update_rejects_non_native_types(monkeypatch) -> None:
    """★ 回写走 ``model_validate`` → 组长的 checkpoint 类型检查会当场生效。

    （如果改用 Pydantic 的 ``model_copy(update=...)``，它**不跑校验器**，
    numpy 之类的值会溜到存 checkpoint 时才炸。）
    """
    import numpy as np

    dn = _data_nodes_or_skip()
    s = _state()
    with pytest.raises(Exception):
        dn._data_update(s, metrics={"bad": np.float64(1.5)})


def test_alarm_codes_come_from_window_data_only(monkeypatch) -> None:
    """★ 报警码只认窗口内的数据，**不读用户入参的 alarm_code**（一个真相来源）。"""
    dn = _data_nodes_or_skip()
    monkeypatch.setattr(dn, "query_scada_telemetry", lambda *a, **k: [
        _row("2026-09-13T00:00:00"), _row("2026-09-13T00:00:05"),
    ])

    out = dn.analyze_node(_state(start="2026-09-13T00:00:00", end="2026-09-13T00:00:05",
                                 alarm_code="FAL-104"))       # 入参有码，窗口里是 NONE

    d = out["data"]
    assert d.quality.status == "OK"
    assert d.quality.total_points == 2
    assert d.alarms.effective == "NONE"
    assert d.alarms.all == []


def test_alarm_codes_three_fields(monkeypatch) -> None:
    """报警码三件套（组长契约）：effective 去重排序 / last 最后一次 / all 列表。"""
    dn = _data_nodes_or_skip()
    monkeypatch.setattr(dn, "query_scada_telemetry", lambda *a, **k: [
        _row("2026-09-13T00:00:00", state="WARNING", alarm_code="TAH-101"),
        _row("2026-09-13T00:00:05", state="WARNING", alarm_code="TAHH-101;VAH-102"),
        _row("2026-09-13T00:00:10", state="WARNING", alarm_code="TAH-101"),
    ])

    d = dn.analyze_node(_state(start="2026-09-13T00:00:00",
                               end="2026-09-13T00:00:10"))["data"]

    assert d.alarms.effective == "TAH-101;TAHH-101;VAH-102"
    assert d.alarms.last == "TAH-101"
    assert d.alarms.all == ["TAH-101", "TAHH-101", "VAH-102"]


def test_judge_flags_returns_codes_and_records_them_per_phase(monkeypatch) -> None:
    """★ 规则判定产出**机器码**：段级写进 ``rule_hits``，窗口级按首次触发顺序去重。"""
    from datetime import datetime, timedelta

    dn = _data_nodes_or_skip()
    base = datetime(2026, 9, 13, 0, 0, 0)
    plan = [("NORMAL", 45.0)] * 3 + [("WARNING", 85.0)] * 3 \
         + [("NORMAL", 45.0)] * 3 + [("WARNING", 85.0)] * 3
    rows = [
        _row((base + timedelta(seconds=5 * i)).isoformat(), state=state, temp_de=temp)
        for i, (state, temp) in enumerate(plan)
    ]
    monkeypatch.setattr(dn, "query_scada_telemetry", lambda *a, **k: rows)

    d = dn.analyze_node(_state(start=base.isoformat(),
                               end=(base + timedelta(seconds=5 * len(plan))).isoformat()))["data"]
    phases = d.metrics["phases"]

    assert [ph["state"] for ph in phases] == ["NORMAL", "WARNING", "NORMAL", "WARNING"]
    assert [ph["rule_hits"] for ph in phases] == [
        [], ["BEARING_TEMP_DE_TRIP"], [], ["BEARING_TEMP_DE_TRIP"],
    ]
    assert d.threshold_flags == ["BEARING_TEMP_DE_TRIP"]         # 窗口级去重


def test_pressure_deviation_rule_fires_and_shares_the_flow_gate(monkeypatch) -> None:
    """★ 出口压力偏离额定值判据（规范 5.1，``|Dev_press| > 20%``）：

    · 泵在转（流量 > 20 m³/h）+ 压头明显不足 → **要报**；
    · 刚停机/尚未建立流动（流量 ≤ 20 m³/h）→ **不报**（此时压力低是必然结果）。
    """
    from datetime import datetime, timedelta

    dn = _data_nodes_or_skip()
    base = datetime(2026, 9, 13, 0, 0, 0)

    def _codes_with(flow_rate: float, press_out: float) -> list[str]:
        monkeypatch.setattr(dn, "query_scada_telemetry", lambda *a, **k: [
            _row((base + timedelta(seconds=5 * i)).isoformat(), state="WARNING",
                 flow_rate=flow_rate, press_out=press_out)
            for i in range(13)          # 13 点 ≥ CV_MIN_POINTS(12)，避免走短段分支
        ])
        return dn.analyze_node(_state(
            start=base.isoformat(),
            end=(base + timedelta(seconds=60)).isoformat(),
        ))["data"].threshold_flags

    running = _codes_with(flow_rate=42.0, press_out=0.131)   # 泵在转，压头不足 → 要报
    stalled = _codes_with(flow_rate=1.0, press_out=0.005)    # 尚未建立流动 → 不报

    assert "PRESS_OUT_DEV" in running
    assert "PRESS_OUT_DEV" not in stalled


def test_evidence_whitelist_covers_material_names() -> None:
    """★ evidence 白名单 = 材料里真实出现过的指标名 + 规则码。"""
    dn = _data_nodes_or_skip()

    allowed = dn._evidence_whitelist(
        overall={"total_points": 3, "phase_count": 1},
        phases=[{"state": "WARNING", "max_temp_de": 85.0, "rule_hits": ["BEARING_TEMP_DE_TRIP"]}],
        rule_codes=["BEARING_TEMP_DE_TRIP"],
    )

    assert {"total_points", "phase_count", "state", "max_temp_de", "rule_hits"} <= allowed
    assert "BEARING_TEMP_DE_TRIP" in allowed
    assert "bearing_temperature" not in allowed          # 编造的名字不在白名单里


def test_semanticize_output_enforces_type_and_cap() -> None:
    """★ LLM 输出契约的结构纪律：type 六选一、描述非空、条数 1~6。"""
    dn = _data_nodes_or_skip()

    with pytest.raises(Exception):                       # type 非法
        dn.DescriptionOut(type="NOT_A_TYPE", description="x")
    with pytest.raises(Exception):                       # 空描述
        dn.DescriptionOut(type="OTHER", description="")
    with pytest.raises(Exception):                       # 超过 6 条
        dn.SemanticizeOutput(descriptions=[
            dn.DescriptionOut(type="OTHER", description="x")
        ] * 7)


def test_summarize_maps_descriptions_and_filters_fabricated_evidence(monkeypatch) -> None:
    """★ descriptions 的映射与过滤：type 原样保留、编造的 evidence 被丢掉、回写不冲字段。

    用假模型替掉真大模型（测试替身只出现在 tests/ 里，靠 monkeypatch 注入）。
    """
    dn = _data_nodes_or_skip()

    monkeypatch.setattr(dn, "query_scada_telemetry", lambda *a, **k: [
        _row("2026-09-13T00:00:00", state="WARNING", temp_de=85.0),
        _row("2026-09-13T00:00:05", state="WARNING", temp_de=85.0),
    ])
    s = _state(start="2026-09-13T00:00:00", end="2026-09-13T00:00:05")
    s = s.model_copy(update=dn.analyze_node(s))          # 先跑 analyze，拿到真实材料

    fake_result = dn.SemanticizeOutput(descriptions=[
        dn.DescriptionOut(type="THRESHOLD", description="驱动端温度越停机线",
                          evidence=["max_temp_de", "bearing_temperature"]),
        dn.DescriptionOut(type="OTHER", description="其余参数平稳", evidence=[]),
    ])

    class _FakeModel:
        """假的聊天模型：with_structured_output 返回一个 Runnable，好接上 prompt | 管道。"""

        def with_structured_output(self, _schema):
            from langchain_core.runnables import RunnableLambda

            return RunnableLambda(lambda _payload: fake_result)

    monkeypatch.setattr(dn, "model", _FakeModel())

    out = dn.summarize_node(s)
    ds = out["data"].descriptions

    assert [d.type for d in ds] == ["THRESHOLD", "OTHER"]
    assert ds[0].evidence == ["max_temp_de"]             # 编造的指标名被过滤
    assert ds[0].description == "驱动端温度越停机线"
    assert list(out) == ["data"]                         # 只回写 data
    assert out["data"].quality.status == "OK"            # analyze 写的字段没被冲掉
    assert out["data"].threshold_flags == s.data.threshold_flags


# =============================================================================
# 6. 挂载：迷你父图（不依赖组长的主图，验证"子图写入 → 父图 state"这条链路）
# =============================================================================


def _mini_parent_graph_or_skip():
    """搭一个 ``START → data → END`` 的迷你父图。

    ★ 为什么不直接用组长的主图：那需要他的 ``build_diagnosis_graph``（还没合过来）。
      迷你父图只依赖公共契约，验证的正是我们负责的那一层：
      子图的写入能不能正确回流到父图、有没有多写别的盒子。
    """
    try:
        from langgraph.graph import StateGraph, START, END

        from src.sub_agents.data_agent.data_graph import data_agent_graph

        parent = StateGraph(DiagnosisState)
        parent.add_node("data", data_agent_graph)
        parent.add_edge(START, "data")
        parent.add_edge("data", END)
        return parent.compile()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"迷你父图不可构建（通常是未配置 DEEPSEEK_API_KEY）：{exc}")


def test_route_b_returns_only_the_data_box() -> None:
    """★ 路线 B 的入口只回写 data。

    主图里 data 与 vision 是**并行分支**：若把整份 state 返回出去，
    就会把并行的 vision 产出一起覆盖掉。所以这里死守"返回值只有 data 一个键"。
    """
    dg = _data_graph_or_skip()
    ret = dg.run_data_agent(_state())        # 无窗口 → 不查库、不调大模型

    assert list(ret) == ["data"]
    assert ret["data"].quality.status == "EMPTY"


def test_mini_parent_graph_returns_only_contract_boxes() -> None:
    """★ 子图挂进父图后，回流的键**不超出**公共契约（没有私有键外流）。"""
    app = _mini_parent_graph_or_skip()
    out = app.invoke(_state().model_dump())          # 无窗口 → 降级，零 API 调用

    leaked = set(out) - TOP_LEVEL_BOXES
    assert not leaked, f"有非契约键回流父图：{sorted(leaked)}"
    assert out["data"].quality.status == "EMPTY"


def test_mini_parent_graph_runs_a_real_window() -> None:
    """★ 真实窗口跑通迷你父图：data 盒子里有指标与描述，顶层只有契约盒子。"""
    import os

    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("需要 DEEPSEEK_API_KEY（本机写在 ~/.bashrc）")

    app = _mini_parent_graph_or_skip()
    out = app.invoke(_state(start="2026-09-13T08:00:00", end="2026-09-13T08:04:55").model_dump())

    if out["data"].quality.status == "EMPTY":
        pytest.skip("数据库窗口内无数据（需要本地 scada_db）")

    assert out["data"].metrics["overall"]["total_points"] > 0
    assert out["data"].descriptions, "正常窗口应当产出一条描述"
    assert out["data"].descriptions[0].description.strip()
    assert not (set(out) - TOP_LEVEL_BOXES)
