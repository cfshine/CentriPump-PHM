"""节点适配函数：从 State 解包参数 → 调用子模块 → 组装更新回写。

本文件只做「搬运」，不含业务算法：
  · 分段与规则判定   → src/sub_agents/data_agent/analyzer.py
  · 通用时序工具     → src/sub_agents/data_agent/timeseries_tools.py
  · 取数（ORM/查询） → src/sub_agents/data_agent/data_tools.py
  · 提示词           → configs/prompts/data_agent.yaml
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from rules.thresholds import DATA_COVERAGE_MIN_PCT, SAMPLE_INTERVAL_SEC
from src.schemas.state import (
    AlarmState,
    DataDescription,
    DataQuality,
    DataSegment,
    DataState,
    DiagnosisState,
    Measurement,
    TelemetryRef,
    node_output,
)
from src.sub_agents.data_agent.analyzer import _judge_flags, _segment_metrics
from src.sub_agents.data_agent.data_tools import _parse_ts, query_scada_telemetry
from src.sub_agents.data_agent.timeseries_tools import (
    _extract_alarm_codes,
    _format_overall,
    _format_phases_for_llm,
    _py,
    render_rule_hits,
)
PROJECT_PATH = Path(__file__).resolve().parents[3]

#: 提示词模板路径（与代码解耦，改提示词不用动 Python）
PROMPT_PATH = PROJECT_PATH / "configs" / "prompts" / "data_agent.yaml"


@lru_cache(maxsize=1)
def _load_prompt_template() -> ChatPromptTemplate:
    """载入 Step 2 的提示词模板。

    返回：
        ``ChatPromptTemplate``（system + user 两条消息）。

    用 lru_cache 缓存：提示词是静态资产，每次语义化都读一遍 YAML 没有意义。
    改完 YAML 重启进程即可生效。
    """
    data = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
    return ChatPromptTemplate.from_messages([
        ("system", data["system"]),
        ("user", data["user"]),
    ])


def _data_update(state: DiagnosisState, **updates) -> dict:
    """把本次要改的字段合并进 ``state.data``，返回**只含 data 一个键**的回写字典。

    参数：
        state:   当前全局状态（读它现有的 data 盒子）。
        updates: 本次要覆盖的 ``DataState`` 字段，如 ``quality=...`` / ``measurements=...``。

    返回：
        ``{"data": DataState}``。
    """
    return {"data": DataState.model_validate({**state.data.model_dump(), **updates})}


# ==================== 大模型输出契约（Pydantic 严格的数据校验）====================

class DescriptionOut(BaseModel):
    """单条数据现象描述（LLM 输出用，字段与 ``DataDescription`` 对齐）。

    参数（由大模型填）：
        kind:        现象类别，六选一（见下列 Literal）
        description: 该现象的自然语言描述；只能描述观测到的现象，禁止故障归因
        evidence:    支撑该描述的指标名或规则码，如 ``['max_temp_de', 'slope_temp_de']``
    """

    kind: Literal["TREND", "ANOMALY", "THRESHOLD", "CORRELATION", "ALARM", "OTHER"] = Field(
        description="该现象的类别。"
    )
    description: str = Field(
        min_length=1,
        description="对数据现象的描述。只描述观测到的数据特征，不得进行故障归因。",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="支撑该描述的指标名或规则码，例如 ['max_temp_de', 'slope_temp_de']。",
    )


class SemanticizeOutput(BaseModel):
    """``summarize_node`` 的 LLM 输出契约：**多条**数据现象描述，按 type 分类。
    """

    descriptions: list[DescriptionOut] = Field(
        min_length=1,
        max_length=6,
        description="1~6 条数据现象描述，按现象分条（不是按阶段分条）。",
    )


_MEASUREMENT_UNITS: dict[str, str] = {
    "duration_sec": "s",
    "data_points": "count",
    "avg_flow": "m³/h",
    "avg_press_out": "MPa",
    "avg_press_in": "MPa",
    "cv_flow": "%",
    "cv_press": "%",
    "slope_temp_de": "℃/min",
    "max_temp_de": "℃",
    "max_temp_nde": "℃",
    "slope_vib_de": "mm/s/min",
    "max_vib_de": "mm/s",
    "max_vib_nde": "mm/s",
    "avg_motor_current": "A",
    "max_motor_current": "A",
    "cv_flow_peak": "%",
    "cv_press_peak": "%",
}

_OVERALL_MEASUREMENT_UNITS: dict[str, str] = {
    "total_points": "count",
    "phase_count": "count",
    "max_temp_de": "℃",
    "max_vib_de": "mm/s",
    "max_temp_nde": "℃",
    "max_vib_nde": "mm/s",
    "avg_flow": "m³/h",
    "min_press_in": "MPa",
}


def _build_phase_records(
    phases: list[dict],
) -> tuple[list[Measurement], list[DataSegment]]:
    """把内部分段字典转换成带单位、带时间窗口的 State 结构。"""

    measurements: list[Measurement] = []
    segments: list[DataSegment] = []

    for index, phase in enumerate(phases):
        segment_key = f"phase:{index}"
        start = str(phase.get("start", ""))
        end = str(phase.get("end", ""))
        segments.append(
            DataSegment(
                index=index,
                state=str(phase.get("state", "")),
                start_time=start,
                end_time=end,
                duration_sec=int(phase.get("duration_sec") or 0),
                data_points=int(phase.get("data_points") or 0),
                rule_hits=list(phase.get("rule_hits") or []),
                ramp_start_time=phase.get("ramp_start_time"),
            )
        )

        for name, unit in _MEASUREMENT_UNITS.items():
            value = phase.get(name)
            if value is None:
                continue
            measurements.append(
                Measurement(
                    name=name,
                    value=float(value),
                    unit=unit,
                    segment=segment_key,
                    start_time=start,
                    end_time=end,
                )
            )

    return measurements, segments


def _build_overall_measurements(
    df: pd.DataFrame,
    phases: list[dict],
    window_start: str,
    window_end: str,
) -> list[Measurement]:
    """生成窗口级指标（安全门禁与提示词共同消费）。"""

    values = {
        "total_points": len(df),
        "phase_count": len(phases),
        "max_temp_de": df["temp_de"].max(),
        "max_vib_de": df["vib_rms_de"].max(),
        "max_temp_nde": df["temp_nde"].max(),
        "max_vib_nde": df["vib_rms_nde"].max(),
        "avg_flow": df["flow_rate"].mean(),
        "min_press_in": df["press_in"].min(),
    }

    measurements: list[Measurement] = []
    for name, unit in _OVERALL_MEASUREMENT_UNITS.items():
        value = _py(values.get(name))
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        measurements.append(
            Measurement(
                name=name,
                value=float(value),
                unit=unit,
                segment="overall",
                start_time=window_start,
                end_time=window_end,
            )
        )
    return measurements


def _phase_views(data: DataState) -> list[dict]:
    """从结构化状态重建分段视图，供既有提示词格式化函数使用。"""

    by_segment: dict[str, dict[str, float]] = {}
    for measurement in data.measurements:
        by_segment.setdefault(measurement.segment, {})[measurement.name] = measurement.value

    views: list[dict] = []
    for segment in data.segments:
        view: dict = {
            "state": segment.state,
            "start": segment.start_time,
            "end": segment.end_time,
            "duration_sec": segment.duration_sec,
            "data_points": segment.data_points,
            "rule_hits": list(segment.rule_hits),
        }
        if segment.ramp_start_time:
            view["ramp_start_time"] = segment.ramp_start_time
        view.update(by_segment.get(f"phase:{segment.index}", {}))
        views.append(view)
    return views


def _overall_view(data: DataState, ctx) -> dict:
    """从结构化状态重建窗口概要视图，供提示词格式化函数使用。"""

    overall: dict = {
        "total_points": data.quality.total_points,
        "window_start": data.telemetry_ref.start_time or ctx.start_time,
        "window_end": data.telemetry_ref.end_time or ctx.end_time,
        "phase_count": len(data.segments),
        "effective_alarm_codes": ";".join(data.alarms.effective),
    }
    if data.segments:
        overall["start_state"] = data.segments[0].state
        overall["end_state"] = data.segments[-1].state
        overall["has_shutdown"] = any(seg.state == "TRIP_SHUTDOWN" for seg in data.segments)

    for measurement in data.measurements:
        if measurement.segment == "overall":
            overall[measurement.name] = measurement.value
    return overall


def _evidence_whitelist(overall: dict, phases: list[dict], rule_codes: list[str]) -> set[str]:
    """材料里真正出现过的指标名与规则码 —— 用来过滤大模型编造的 evidence。

    参数：
        overall:    窗口级指标视图（每个指标名对应一个数值）
        phases:     分段指标视图（每段的指标名与数值）
        rule_codes: ``data.threshold_flags``（规则码本身也可作为证据）

    返回：
        可接受的 evidence 名字集合。
    """
    allowed = set(overall)
    for ph in phases:
        allowed.update(ph)
    allowed.update(rule_codes)
    return allowed


# ==================== 节点 1: 取数 + 确定性计算 ====================

@node_output("analyze")
def analyze_node(state: DiagnosisState) -> dict:
    """节点1：取时间窗口内的原始遥测，并做确定性计算（纯 pandas，无大模型）。

    参数：
        state: 全局状态；读 ``state.context`` 的 device_id / start_time / end_time / trace_id。

    返回：
        ``{"data": DataState}``，三种出口：
          正常         → quality=OK，measurements / segments / threshold_flags / alarms 全部填好
          没有时间窗口 → quality=EMPTY + reason（**不查库**）
          窗口内无数据 → quality=EMPTY + reason
    """
    ctx = state.context
    telemetry_ref = TelemetryRef(
        source=ctx.telemetry_uri or "scada_db.scada_telemetry",
        trace_id=ctx.trace_id,
        device_id=ctx.device_id,
        start_time=ctx.start_time,
        end_time=ctx.end_time,
    )

    if not (ctx.start_time.strip() and ctx.end_time.strip()):
        print("[analyze] 未提供时间窗口 → 跳过取数与计算（本次不做时序分析）")
        return _data_update(
            state,
            telemetry_ref=telemetry_ref,
            quality=DataQuality(status="EMPTY", total_points=0,
                                reason="未提供时间窗口：本次未做时序分析"),
            threshold_flags=[], alarms=AlarmState(),
        )

    if ctx.analysis_mode == "offline" and not ctx.telemetry_uri:
        raise ValueError("离线时序分析必须显式提供 telemetry_uri CSV 路径")

    print(f"[analyze] 拉取 {ctx.device_id} 在 {ctx.start_time} ~ {ctx.end_time} 的数据")
    raw = query_scada_telemetry(ctx.device_id, ctx.start_time, ctx.end_time,
                               csv_path=ctx.telemetry_uri)
    print(f"[analyze] 获取到 {len(raw)} 条记录")

    df = pd.DataFrame(raw)
    if df.empty:
        return _data_update(
            state,
            telemetry_ref=telemetry_ref,
            quality=DataQuality(status="EMPTY", total_points=0,
                                reason="指定时间窗口内无 SCADA 记录"),
            threshold_flags=[], alarms=AlarmState(),
        )

    print("[analyze] 开始确定性计算（分段统计 + 规则判定）")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # —— 1. 窗口内的报警码三件套（只认数据，不认用户入参）
    effective, last, all_codes = _extract_alarm_codes(df)

    # —— 2. 分段：按 operating_state 变化切段
    df['_seg_id'] = (df['operating_state'] != df['operating_state'].shift()).cumsum()

    phases = []
    for _, seg in df.groupby('_seg_id', sort=True):
        ph = _segment_metrics(seg)
        ph["state"] = seg['operating_state'].iloc[0]
        phases.append(ph)

    # —— 3. 分段规则判定：命中的**机器码**写回各段 rule_hits，并返回窗口级去重码
    rule_codes = _judge_flags(phases)

    # —— 4. 结构化指标：窗口级 + 分段级，全部带单位与时间窗口 ——
    window_start = df['timestamp'].iloc[0].isoformat()
    window_end = df['timestamp'].iloc[-1].isoformat()
    phase_measurements, segments = _build_phase_records(phases)
    measurements = phase_measurements + _build_overall_measurements(
        df, phases, window_start, window_end
    )

    # —— 5. 数据完整性：窗口里本该有多少点 vs 实际拿到多少点 ——
    window_sec = (_parse_ts(ctx.end_time) - _parse_ts(ctx.start_time)).total_seconds()
    expected_points = int(window_sec // SAMPLE_INTERVAL_SEC) + 1
    coverage_pct = len(df) / expected_points * 100 if expected_points > 0 else 100.0
    if coverage_pct < DATA_COVERAGE_MIN_PCT:
        quality = DataQuality(
            status="PARTIAL", total_points=int(len(df)),
            reason=f"窗口内缺失约 {100 - coverage_pct:.0f}% 的数据"
                   f"（理论 {expected_points} 点，实际 {len(df)} 点）",
        )
    else:
        quality = DataQuality(status="OK", total_points=int(len(df)), reason="")

    print(f"[analyze] 识别 {len(phases)} 个阶段，命中 {len(rule_codes)} 种规则: {rule_codes}")
    print(f"[analyze] 窗口内报警码: {effective}")
    print(f"[analyze] 数据质量: {quality.status}（覆盖率 {coverage_pct:.0f}%）")

    return _data_update(
        state,
        telemetry_ref=telemetry_ref,
        quality=quality,
        measurements=measurements,
        segments=segments,
        threshold_flags=rule_codes,
        alarms=AlarmState(effective=effective, last=last, all=all_codes),
    )


# ==================== 节点 2: LLM 语义化 ====================

@node_output("summarize")
def summarize_node(state: DiagnosisState) -> dict:
    """节点2：把 Python 算好的数字交给大模型，产出数据现象描述（写进 ``data.descriptions``）。

    参数：
        state: 全局状态；读 ``state.context``（提示词用）与 ``state.data``（材料）。

    返回：
        正常          → ``{"data": DataState}``，``descriptions`` 里装 1~6 条按 kind 分类的现象描述
        没窗口 / 没数据 → ``{}``（什么都不写，也**不调大模型**）
    """
    ctx = state.context
    if not (ctx.start_time.strip() and ctx.end_time.strip()):
        print("[summarize] 未提供时间窗口 → 跳过语义化（不调用大模型）")
        return {}

    data = state.data
    if data.quality.status == "EMPTY":
        print(f"[summarize] 数据质量 {data.quality.status} → 跳过语义化（不调用大模型）")
        return {}

    if ctx.analysis_mode == "offline":
        descriptions = [DataDescription(
            kind="THRESHOLD",
            description=f"离线规则计算命中 {code}；未调用语言模型。",
            evidence=[code],
        ) for code in data.threshold_flags]
        if not descriptions:
            descriptions = [DataDescription(kind="OTHER", description="离线规则计算未命中阈值；未调用语言模型。")]
        return _data_update(state, descriptions=descriptions)

    phases = _phase_views(data)
    overall = _overall_view(data, ctx)

    # 阈值命中材料：把机器码渲染成中文事实句给大模型看（渲染只发生在这里，不进 state）
    rule_hits_text = render_rule_hits(phases)

    prompt = _load_prompt_template()
    from src.utils.llm_client import model
    chain = prompt | model.with_structured_output(SemanticizeOutput)

    result: SemanticizeOutput = chain.invoke({
        "device_id": ctx.device_id,
        "time_window": f"{ctx.start_time} ~ {ctx.end_time}",
        "overall_summary": _format_overall(overall),
        "phases_text": _format_phases_for_llm(phases),
        "flags": rule_hits_text,
    })

    if result is None:
        raise RuntimeError(
            "大模型结构化输出解析失败（返回 None）；通常是输出被 max_tokens 截断，"
            "请调大 src/utils/llm_client.py 里的 max_tokens，或收紧提示词的长度要求"
        )

    # 映射成大模型的返回值 → 过滤掉编造的 evidence
    allowed_evidence = _evidence_whitelist(overall, phases, data.threshold_flags)
    descriptions = [
        DataDescription(
            kind=d.kind,
            description=d.description,
            evidence=[e for e in d.evidence if e in allowed_evidence],
        )
        for d in result.descriptions
    ]
    if any(not d.description.strip() for d in descriptions):
        raise RuntimeError("大模型返回了空描述；请检查提示词或 max_tokens 设置")

    # 预警线：按**事件数**算（口径见 README；这条只是日志提醒，不影响断言）
    state_switches = max(0, len(phases) - 1)
    alarm_code_count = len(data.alarms.effective)
    warn_threshold = min(1000, 500 + 40 * state_switches + 20 * alarm_code_count)
    total_len = sum(len(d.description) for d in descriptions)
    if total_len > warn_threshold:
        print(
            f"[summarize] ⚠ LLM 描述较长 ({total_len} 字 / {len(descriptions)} 条，"
            f"{state_switches} 次状态切换 + {alarm_code_count} 种报警码 → 预警线 {warn_threshold})，"
            f"请检查是否违反'连续同状态段合并'原则"
        )

    return _data_update(state, descriptions=descriptions)
