"""节点适配函数：从 State 解包参数 → 调用子模块 → 组装更新回写。

本文件只做「搬运」，不含业务算法：
  · 分段与规则判定   → src/sub_agents/data_agent/analyzer.py
  · 通用时序工具     → src/sub_agents/data_agent/timeseries_tools.py
  · 取数（ORM/查询） → src/sub_agents/data_agent/data_tools.py
  · 提示词           → configs/prompts/data_agent.yaml

★ 2026-09-20 适配组长的 9 盒子契约：
    · 输入不再在顶层，改读 ``state.context``（device_id / start_time / end_time / trace_id）；
    · 产出不再写顶层字段，改写成 ``state.data``（DataState）；
    · 回写统一走 ``_data_update`` —— 必须带上盒子里其他字段，且必须走校验；
    · **只回写 data 一个顶层盒子**：主图里 data 与 vision 是并行分支，
      回写整份 state 会覆盖 vision 的产出。
"""
from functools import lru_cache
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.schemas.state import (
    AlarmState,
    DataDescription,
    DataQuality,
    DataState,
    DiagnosisState,
    TelemetryRef,
)
from src.sub_agents.data_agent.analyzer import _judge_flags, _segment_metrics
from src.sub_agents.data_agent.data_tools import query_scada_telemetry
from src.sub_agents.data_agent.timeseries_tools import (
    _extract_alarm_codes,
    _format_overall,
    _format_phases_for_llm,
    _py,
    render_rule_hits,
)
from src.utils.config_loader import PROJECT_PATH
from src.utils.llm_client import model

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
        updates: 本次要覆盖的 ``DataState`` 字段，如 ``quality=...`` / ``metrics=...``。

    返回：
        ``{"data": DataState}``。

    两个必须这么写的理由（都是实测出来的）：
        1. LangGraph 用返回值**替换整个盒子**：只返回部分字段，会把盒子里其他字段
           （比如 analyze 刚写好的 metrics）一起冲掉。所以先摊平现有内容再覆盖。
        2. 不能用 Pydantic 的 ``model_copy(update=...)`` —— 它**不跑校验器**，会绕过
           ``StateModel`` 的 checkpoint 类型检查（只允许精确的
           str/int/float/bool/None/list/dict）。这里用 ``model_validate`` 重新构造，
           校验当场发生。
    """
    return {"data": DataState.model_validate({**state.data.model_dump(), **updates})}


# ==================== 大模型输出契约（Pydantic 严格的数据校验）====================

class DescriptionOut(BaseModel):
    """单条数据现象描述（LLM 输出用，字段与组长的 ``DataDescription`` 对齐）。

    参数（由大模型填）：
        type:        现象类别，六选一（见下列 Literal）
        description: 该现象的自然语言描述；只能描述观测到的现象，禁止故障归因
        evidence:    支撑该描述的指标名或规则码，如 ``['max_temp_de', 'slope_temp_de']``

    说明：
        这里是"大模型返回值"的契约；节点会把它映射成组长的 ``DataDescription``，
        并按白名单过滤 evidence（编造的指标名会被丢掉）。
    """

    type: Literal["TREND", "ANOMALY", "THRESHOLD", "CORRELATION", "ALARM", "OTHER"] = Field(
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

    ★ 为什么是列表而不是一段话（2026-09-20）：
      组长的 ``DataState.descriptions`` 是 ``list[DataDescription]``，
      下游（Step 4 RAG / Step 7 报告）要按类别取用；
      一段整话既没法检索、也没法按类型筛选。
      条数上限定 6 条是为了防大模型"刷条数"把描述拆成流水账。
    """

    descriptions: list[DescriptionOut] = Field(
        min_length=1,
        max_length=6,
        description="1~6 条数据现象描述，按现象分条（不是按阶段分条）。",
    )


def _evidence_whitelist(overall: dict, phases: list[dict], rule_codes: list[str]) -> set[str]:
    """材料里真正出现过的指标名与规则码 —— 用来过滤大模型编造的 evidence。

    参数：
        overall:    ``data.metrics["overall"]``（窗口概要与它的键名）
        phases:     ``data.metrics["phases"]``（每段的指标键名）
        rule_codes: ``data.threshold_flags``（规则码本身也可作为证据）

    返回：
        可接受的 evidence 名字集合。

    为什么要过滤：
        组长的契约要求 evidence 能回溯到真实数据。大模型偶尔会写
        "bearing_temperature" 这种看起来合理、但材料里根本不存在的名字；
        与其在报告里留下无法溯源的证据，不如在入口就把它减掉。
    """
    allowed = set(overall)
    for ph in phases:
        allowed.update(ph)
    allowed.update(rule_codes)
    return allowed


# ==================== 节点 1: 取数 + 确定性计算 ====================

def analyze_node(state: DiagnosisState) -> dict:
    """节点1：取时间窗口内的原始遥测，并做确定性计算（纯 pandas，无大模型）。

    参数：
        state: 全局状态；读 ``state.context`` 的 device_id / start_time / end_time / trace_id。

    返回：
        ``{"data": DataState}``，三种出口：
          正常         → quality=OK，metrics / threshold_flags / alarms / telemetry_ref 全部填好
          没有时间窗口 → quality=EMPTY + reason（**不查库**）
          窗口内无数据 → quality=EMPTY + reason
    """
    ctx = state.context
    telemetry_ref = TelemetryRef(
        source="scada_db.scada_telemetry",
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
            metrics={}, threshold_flags=[], alarms=AlarmState(),
        )

    print(f"[analyze] 拉取 {ctx.device_id} 在 {ctx.start_time} ~ {ctx.end_time} 的数据")
    raw = query_scada_telemetry(ctx.device_id, ctx.start_time, ctx.end_time)
    print(f"[analyze] 获取到 {len(raw)} 条记录")

    df = pd.DataFrame(raw)
    if df.empty:
        return _data_update(
            state,
            telemetry_ref=telemetry_ref,
            quality=DataQuality(status="EMPTY", total_points=0,
                                reason="指定时间窗口内无 SCADA 记录"),
            metrics={}, threshold_flags=[], alarms=AlarmState(),
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

    # —— 4. 全局概要（报警码不在这里，改由 data.alarms 单一出处）
    overall = {
        "total_points": int(len(df)),
        "window_start": df['timestamp'].iloc[0].isoformat(),
        "window_end": df['timestamp'].iloc[-1].isoformat(),
        "start_state": df['operating_state'].iloc[0],
        "end_state": df['operating_state'].iloc[-1],
        "has_shutdown": bool((df['operating_state'] == 'TRIP_SHUTDOWN').any()),
        "phase_count": len(phases),
        "max_temp_de_overall": _py(df['temp_de'].max(), 1),
        "max_vib_de_overall": _py(df['vib_rms_de'].max(), 2),
        "max_temp_nde_overall": _py(df['temp_nde'].max(), 1),
        "max_vib_nde_overall": _py(df['vib_rms_nde'].max(), 2),
    }

    print(f"[analyze] 识别 {len(phases)} 个阶段，命中 {len(rule_codes)} 种规则: {rule_codes}")
    print(f"[analyze] 窗口内报警码: {effective}")

    return _data_update(
        state,
        telemetry_ref=telemetry_ref,
        quality=DataQuality(status="OK", total_points=int(len(df)), reason=""),
        metrics={"overall": overall, "phases": phases},
        threshold_flags=rule_codes,
        alarms=AlarmState(effective=effective, last=last, all=all_codes),
    )


# ==================== 节点 2: LLM 语义化 ====================

def summarize_node(state: DiagnosisState) -> dict:
    """节点2：把 Python 算好的数字交给大模型，产出数据现象描述（写进 ``data.descriptions``）。

    参数：
        state: 全局状态；读 ``state.context``（提示词用）与 ``state.data``（材料）。

    返回：
        正常          → ``{"data": DataState}``，``descriptions`` 里装 1~6 条按 type 分类的现象描述
        没窗口 / 没数据 → ``{}``（什么都不写，也**不调大模型**）

    ★ 时间窗口的判断与 ``analyze_node`` 里那一行必须**完全一致**，否则会出现
      "取数跳过了、语义化却照调大模型"这种半截状态。
    ★ 没有数据时不写描述：降级原因已经在 ``data.quality.reason`` 里，
      再生成一段"未做分析"的话只会污染 descriptions。
    ★ evidence 会按白名单过滤（只留材料里出现过的指标名/规则码），
      防止大模型编造无法溯源的证据。
    """
    ctx = state.context
    if not (ctx.start_time.strip() and ctx.end_time.strip()):
        print("[summarize] 未提供时间窗口 → 跳过语义化（不调用大模型）")
        return {}

    data = state.data
    if data.quality.status == "EMPTY":
        print(f"[summarize] 数据质量 {data.quality.status} → 跳过语义化（不调用大模型）")
        return {}

    phases = data.metrics.get("phases", [])
    overall = data.metrics.get("overall", {})

    # 阈值命中材料：把机器码渲染成中文事实句给大模型看（渲染只发生在这里，不进 state）
    rule_hits_text = render_rule_hits(phases)

    prompt = _load_prompt_template()
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

    # 映射成大模型的返回值 → 组长的 DataDescription，并过滤掉编造的 evidence
    allowed_evidence = _evidence_whitelist(overall, phases, data.threshold_flags)
    descriptions = [
        DataDescription(
            type=d.type,
            description=d.description,
            evidence=[e for e in d.evidence if e in allowed_evidence],
        )
        for d in result.descriptions
    ]
    if any(not d.description.strip() for d in descriptions):
        raise RuntimeError("大模型返回了空描述；请检查提示词或 max_tokens 设置")

    # 预警线：按**事件数**算（口径见 README；这条只是日志提醒，不影响断言）
    state_switches = max(0, len(phases) - 1)
    codes = data.alarms.effective
    alarm_code_count = 0 if codes in ("", "NONE") else len([c for c in codes.split(";") if c.strip()])
    warn_threshold = min(1000, 500 + 40 * state_switches + 20 * alarm_code_count)
    total_len = sum(len(d.description) for d in descriptions)
    if total_len > warn_threshold:
        print(
            f"[summarize] ⚠ LLM 描述较长 ({total_len} 字 / {len(descriptions)} 条，"
            f"{state_switches} 次状态切换 + {alarm_code_count} 种报警码 → 预警线 {warn_threshold})，"
            f"请检查是否违反'连续同状态段合并'原则"
        )

    return _data_update(state, descriptions=descriptions)
