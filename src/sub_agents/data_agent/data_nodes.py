"""节点适配函数：从 State 解包参数 → 调用子模块 → 组装更新回写。

本文件只做「搬运」，不含业务算法：
  · 分段与规则判定   → src/sub_agents/data_agent/analyzer.py
  · 通用时序工具     → src/sub_agents/data_agent/timeseries_tools.py
  · 取数（ORM/查询） → src/sub_agents/data_agent/repository.py
  · 提示词           → configs/prompts/data_agent.yaml
"""
from functools import lru_cache

import pandas as pd
import yaml
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.sub_agents.data_agent.data_tools import query_scada_telemetry, query_alarm_events
from src.sub_agents.data_agent.data_state import DataAgentState
from src.utils.llm_client import model
from src.utils.config_loader import PROJECT_PATH
from src.sub_agents.data_agent.timeseries_tools import _py, _extract_alarm_codes, _format_alarm_events_for_llm, _format_overall, _format_phases_for_llm
from src.sub_agents.data_agent.analyzer import _segment_metrics, _judge_flags

#: 提示词模板路径（与代码解耦，改提示词不用动 Python）
PROMPT_PATH = PROJECT_PATH / "configs" / "prompts" / "data_agent.yaml"


@lru_cache(maxsize=1)
def _load_prompt_template() -> ChatPromptTemplate:
    """载入 Step 2 的提示词模板。

    用 lru_cache 缓存：提示词是静态资产，每次语义化都读一遍 YAML 没有意义。
    改完 YAML 重启进程即可生效。
    """
    data = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
    return ChatPromptTemplate.from_messages([
        ("system", data["system"]),
        ("user", data["user"]),
    ])



# ==================== 大模型输出契约（Pydantic 严格的数据校验）====================

class SemanticizeOutput(BaseModel):
    """semanticize_node 的 LLM 输出契约。
    区别（实测会踩的坑）：
        JsonOutputParser   → LLM 漏字段返回 {}、类型写错（如 rag_search_queries
                             返回字符串而非列表）都**静默通过**，下游 for 循环
                             逐字符迭代，变成难查的隐性 bug。
        with_structured_output → 上面两种情况直接抛 ValidationError，
                             出错点明确，能被测试立刻发现。

    用它之后，system prompt 里**不再需要手写 JSON 格式说明**，
    schema 由 LangChain 自动注入，prompt 只讲业务规则。
    """

    llm_description: str = Field(
        ...,
        description="按时间顺序的工况叙事描述。连续同状态段要合并，"
                    "状态切换时刻必须提及，只写峰值与触发码、不复述数值表。"
                    "长度与「发生的事件数」成比例：24 小时窗口不超过 1000 字。",
    )
    basic_judgment: str = Field(
        ...,
        description="1~3 句基础判断，指出关键异常时刻与所处阶段",
    )
    rag_search_queries: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3~5 条供 Step 4 RAG 检索用的检索词",
    )


# ==================== 节点 1: 拉取数据 ====================

def _has_window(state: DataAgentState) -> bool:
    """判断本次调用**是否提供了时间窗口**（软降级的唯一判据）。

    参数：
        state: 子图状态；读 ``start_time`` 与 ``end_time``。

    返回：
        True  = 起止时间都非空 → 正常做时序分析。
        False = 缺任一个（或都是空串）→ 三个节点统一按"本次不做时序分析"处理。

    为什么用这一个判据、而不是每处各判一次：
        三个节点（取数 / 计算 / 语义化）必须**做出同样的判断**，否则会出现
        "取数跳过了、语义化却照调大模型"这种半截状态。集中成一个函数最不容易漂移。
    """
    return bool(
        str(state.get("start_time") or "").strip()
        and str(state.get("end_time") or "").strip()
    )


def fetch_data_node(state: DataAgentState):
    """节点1：取时间窗口内的原始遥测。

    参数：
        state: 子图状态；读 ``device_id`` / ``start_time`` / ``end_time``。

    返回：
        正常 → ``{"raw_telemetry_data": [ ...每帧一条 dict... ]}``（私有字段，不外流）
        软降级 → ``{"raw_telemetry_data": []}`` —— **不查库**

    ★ 软降级（2026-09-17 用户拍板）：
        未提供时间窗口时**直接返回空列表，不做任何数据库查询**。
        以前空窗口会一路走到 ``_parse_ts`` 抛 ``ValueError: Invalid isoformat string: ''``，
        把整张 LangGraph 掀翻（实测：只给图片路径的输入会让主图整体失败，
        连与 Step 2 完全无关的 Step 3 都轮不到执行）。
        现在它安全退出，把"这次没做时序分析"交给下游两个节点显式表达。
    """
    if not _has_window(state):
        print("[Node 1] 未提供时间窗口 → 跳过时序取数（本次不做时序分析）")
        return {"raw_telemetry_data": []}

    print(f"[Node 1] 拉取 {state['device_id']} 在 {state['start_time']} ~ {state['end_time']} 的数据")
    data = query_scada_telemetry(
        state["device_id"], state["start_time"], state["end_time"]
    )
    print(f"[Node 1] 获取到 {len(data)} 条记录")
    return {"raw_telemetry_data": data}


# ==================== 节点 2: 分段确定性计算 ====================

def calculate_metrics_node(state: DataAgentState):
    """节点2：分段统计 + 分段告警判定（纯 pandas，无数据库/大模型依赖）。

    参数：
        state: 子图状态；读 ``raw_telemetry_data`` 与 ``alarm_code``。

    返回：
        正常 → 五个公共字段（``calculated_metrics`` / ``threshold_flags`` /
               ``effective_alarm_codes`` / ``last_alarm_codes`` / ``all_alarm_codes_in_window``）
        无数据 → 空指标 + 一条"**未提供时间窗口：本次未做时序分析**"告警
    """
    if not _has_window(state):
        print("[Node 2] 未提供时间窗口 → 跳过确定性计算（本次不做时序分析）")
        return {
            "calculated_metrics": {},
            "threshold_flags": ["未提供时间窗口：本次未做时序分析"],
            "effective_alarm_codes": state.get("alarm_code", "") or "NONE",
            "last_alarm_codes": "NONE",
            "all_alarm_codes_in_window": [],
        }

    print("[Node 2] 开始确定性计算（分段统计）")

    df = pd.DataFrame(state["raw_telemetry_data"])
    if df.empty:
        return {
            "calculated_metrics": {},
            "threshold_flags": ["无数据：指定时间窗口内无 SCADA 记录"],
            "effective_alarm_codes": state.get("alarm_code", "") or "NONE",
        }

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 提取报警码（返回三元组）
    effective, last, all_codes = _extract_alarm_codes(
        df, fallback_alarm_code=state.get("alarm_code", "")
    )

    # —— 1. 分段：按 operating_state 变化切段
    df['_seg_id'] = (df['operating_state'] != df['operating_state'].shift()).cumsum()

    phases = []
    for _, seg in df.groupby('_seg_id', sort=True):
        ph = _segment_metrics(seg)
        ph["state"] = seg['operating_state'].iloc[0]
        phases.append(ph)

    # —— 2. 分段阈值判定
    flags = _judge_flags(phases)

    # —— 3. 全局概要
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
        "effective_alarm_codes": effective,
        "last_alarm_codes": last,
        "all_alarm_codes_in_window": all_codes,
    }

    metrics = {"overall": overall, "phases": phases}

    print(f"[Node 2] 识别 {len(phases)} 个阶段，触发 {len(flags)} 条阈值告警")
    print(f"[Node 2] 全量报警码: {effective}")
    print(f"[Node 2] 最后一次报警码: {last}")

    return {
        "calculated_metrics": metrics,
        "threshold_flags": flags,
        "effective_alarm_codes": effective,      
        "last_alarm_codes": last,                
        "all_alarm_codes_in_window": all_codes,  
    }


# ==================== 节点 3: LLM 语义化 ====================
def semanticize_node(state: DataAgentState):
    """节点3：LLM 语义化 + 按需核验报警事件。

    参数：
        state: 子图状态；读 ``calculated_metrics`` / ``threshold_flags`` /
               ``all_alarm_codes_in_window`` / ``device_id`` / 时间窗口。

    返回：
        正常 → ``llm_description`` / ``basic_judgment`` / ``rag_search_queries``
               + 私有字段 ``alarm_events``（不回流的报警段）
        软降级 → 同样的四个键，但内容是**确定性模板**，且**不调用大模型**

    ★ 软降级：
        未提供时间窗口时，既不核验报警事件（那也要窗口）、也不调大模型
        返回的文本明确写出"本次未做时序分析"，让 Step 4/7 知道这是**没做**，
        而不是"做了但没发现异常"。
    """
    if not _has_window(state):
        print("[Node 3] 未提供时间窗口 → 跳过语义化（不调用大模型）")
        return {
            "llm_description": "未提供时间窗口：本次未做时序分析，无工况描述。",
            "basic_judgment": "未提供时间窗口，Step 2 未执行时序分析（未取数、未调用大模型）。",
            "rag_search_queries": [],
            "alarm_events": [],
        }

    print("[Node 3] 调用大模型生成语义化描述")

    metrics = state.get("calculated_metrics", {})
    phases = metrics.get("phases", [])
    overall = metrics.get("overall", {})
    has_alarm = bool(state.get("all_alarm_codes_in_window"))

    # —— 判断是否需要核验报警事件
    alarm_events = []
    if has_alarm:
        print("[Node 3] 检测到报警码，调用 query_alarm_events 核验报警时刻")
        try:
            alarm_events = query_alarm_events(
                state["device_id"], state["start_time"], state["end_time"]
            )
            print(f"[Node 3] 核验到 {len(alarm_events)} 条报警事件")
        except Exception as e:
            print(f"[Node 3] 报警事件查询失败: {e}")

    # —— 把报警事件格式化为 LLM 输入文本
    alarm_events_text = _format_alarm_events_for_llm(alarm_events)

    # —— 构造 Prompt（报警码仍不直接注入系统提示，但事件数据作为"用户材料"传入）
    prompt = _load_prompt_template()

    chain = prompt | model.with_structured_output(SemanticizeOutput)

    result: SemanticizeOutput = chain.invoke({
        "device_id": state["device_id"],
        "time_window": f"{state['start_time']} ~ {state['end_time']}",
        "overall_summary": _format_overall(overall),
        "phases_text": _format_phases_for_llm(phases),
        "alarm_events_section": alarm_events_text,   # 为空时返回 ""
        "flags": "\n".join(f"- {f}" for f in state["threshold_flags"]) or "无",
    })

    # ★ with_structured_output 在「解析失败」时返回 None（最常见原因是输出被
    #   max_tokens 截断）。这里给一句能直接定位问题的报错，而不是让下游抛出
    #   莫名其妙的 'NoneType' object has no attribute 'llm_description'。
    if result is None:
        raise RuntimeError(
            "大模型结构化输出解析失败（返回 None）；通常是输出被 max_tokens 截断，"
            "请调大 src/utils/llm_client.py 里的 max_tokens，或收紧提示词的长度要求"
        )

    # 软预警：超长时打印日志，不截断（保留完整信息）
    desc = result.llm_description or ""

    # 预警线：按**事件数**算，而不是阶段数。
    # ★ 阶段数分级太粗 —— 一个"只有 2 段、但报警码升级了 19 次"的窗口，
    #   事件一点不少，却会被 500 字的固定线卡住（实测 A2 窗口写 525 字被误报）。
    #   事件数 = 状态切换次数 + 报警段数，二者都是"这段窗口里发生了多少事"的直接度量。
    #   实测标定：A2 窗口（1 次切换 + 18 段报警）同一份输入多次调用写 545~650 字
    #            （temperature=0 也有 ±50 字的波动）；24h 窗口（9 次切换 + 30 段）写 707~810 字。
    #   取 500 + 40×切换 + 10×段 —— 两者分别得 720 与 1160(封顶 1000)，
    #   都按实测**上沿**留了余量，不会因为几十字的正常波动就误报。
    phase_count = len(phases)
    state_switches = max(0, phase_count - 1)
    alarm_runs = len(alarm_events)
    warn_threshold = min(1000, 500 + 40 * state_switches + 10 * alarm_runs)
    if len(desc) > warn_threshold:
        print(
            f"[Node 3] ⚠ LLM 描述较长 ({len(desc)} 字，"
            f"{state_switches} 次状态切换 + {alarm_runs} 段报警 → 预警线 {warn_threshold})，"
            f"请检查是否违反'连续同状态段合并'原则"
        )

    return {
        "llm_description": result.llm_description,
        "basic_judgment": result.basic_judgment,
        "rag_search_queries": result.rag_search_queries,
        "alarm_events": alarm_events,
    }