# src/sub_agents/data_agent/timeseries_tools.py
import pandas as pd
import numpy as np
from scipy import stats

# ==================== 通用工具 ====================

# ★ 全部数值判据统一来自 rules/thresholds.py（唯一出处）。
#   这里用别名导入，保持原有下划线变量名不变 —— 下游（analyzer.py）引用它们的
#   代码一行都不用改，而阈值本身从此只有一份定义。
from rules.thresholds import (
    ACTIVE_STATES as _ACTIVE_STATES,
    FLOW_ACTIVE_THRESHOLD_M3H as _FLOW_ACTIVE_THRESHOLD,
    SLOPE_WINDOW_POINTS,
    TEMP_SLOPE_ACTIVATION_C as _TEMP_SLOPE_ACTIVATION,
    TEMP_SLOPE_MIN_DURATION_SEC as _TEMP_SLOPE_MIN_DURATION_SEC,
    TEMP_SLOPE_MIN_POINTS as _TEMP_SLOPE_MIN_POINTS,
    TEMP_SLOPE_SHARP,
    TEMP_WARN_C,
)

def _py(v, ndigits=None):
    """将 numpy 类型统一转为 Python 原生类型，避免 JSON 序列化报错。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (np.floating, float)):
        v = float(v)
    elif isinstance(v, (np.integer, int)):
        v = int(v)
    if ndigits is not None and isinstance(v, float):
        v = round(v, ndigits)
    return v


def _safe_slope(minutes: pd.Series, values: pd.Series) -> float:
    """线性回归斜率；点不足 2 或时间跨度为 0 时返回 0.0"""
    if len(minutes) < 2:
        return 0.0
    if minutes.max() - minutes.min() <= 0:
        return 0.0
    try:
        return float(stats.linregress(minutes, values).slope)
    except Exception:
        return 0.0


def _cv(series: pd.Series) -> float:
    """变异系数 CV = std/mean*100；均值为 0、点数不足、NaN 时返回 0。"""
    if len(series) < 2:          # 单点没有离散度概念
        return 0.0
    m = series.mean()
    if pd.isna(m) or m == 0:
        return 0.0
    s = series.std()
    if pd.isna(s):               # std 可能是 NaN
        return 0.0
    return float(s / m * 100)

def _ge(v, threshold) -> bool:
    """None 安全的 >= 比较：None 视为不满足"""
    return v is not None and v >= threshold

def _gt(v, threshold) -> bool:
    return v is not None and v > threshold

def _lt(v, threshold) -> bool:
    """None 安全的 < 比较"""
    return v is not None and v < threshold


def _is_valid_alarm(code) -> bool:
    """判断一条 alarm_code 是否为有效报警（过滤 NONE / 空 / null）"""
    return isinstance(code, str) and code.strip() not in ("", "NONE", "None", "null")


def _extract_alarm_codes(df: pd.DataFrame, fallback_alarm_code: str = "") -> tuple[str, str, list[str]]:
    """
    从 DataFrame 中提取有效报警码。
    
    返回三元组：
      - effective_alarm_codes: 窗口内所有出现过的报警码去重排序（分号连接）
      - last_alarm_codes: 窗口内最后一次非 NONE 的报警码组合
      - all_codes_list: 去重后的列表形式
    """
    valid_rows = df[df['alarm_code'].apply(_is_valid_alarm)]
    
    if valid_rows.empty:
        last = (fallback_alarm_code or "NONE").strip()
        return last, last, []
    
    # 收集窗口内出现过的所有报警码
    all_codes = set()
    for codes_str in valid_rows['alarm_code']:
        for code in codes_str.split(";"):
            code = code.strip()
            if code:
                all_codes.add(code)
    all_codes_list = sorted(all_codes)
    
    # 最后一次非 NONE 组合
    last_alarm_codes = valid_rows['alarm_code'].iloc[-1].strip()
    
    # 全量去重排序（分号连接）
    effective = ";".join(all_codes_list)
    
    return effective, last_alarm_codes, all_codes_list


def _detect_inflection(seg: pd.DataFrame) -> dict:
    """
    检测段内关键指标的拐点（首次突破阈值时刻）。
    
    返回：
      - inflection_time: 温度首次 >= 70℃ 的时刻（无则 None）
      - inflection_value: 该时刻的温度值（无则 None）
      - ramp_start_time: 温度斜率首次 >= 0.8 的时刻（无则 None）
    """
    result = {
        "inflection_time": None,
        "inflection_value": None,
        "ramp_start_time": None,
    }
    
    if len(seg) < 2:
        return result
    
    # 检测温度首次达到预警线的时刻（阈值见 rules/thresholds.py）
    over_threshold = seg[seg['temp_de'] >= TEMP_WARN_C]
    if not over_threshold.empty:
        first = over_threshold.iloc[0]
        result["inflection_time"] = first['timestamp'].isoformat()
        result["inflection_value"] = float(first['temp_de'])
    
    # 检测斜率首次达到"急剧恶化"的时刻（用 1 分钟滑窗斜率）
    if len(seg) >= SLOPE_WINDOW_POINTS:  # 至少 1 分钟数据
        t0 = seg['timestamp'].min()
        minutes = (seg['timestamp'] - t0).dt.total_seconds() / 60
        # 滑动窗口回归（每 1 分钟 = SLOPE_WINDOW_POINTS 点）
        rolling_slopes = []
        for i in range(SLOPE_WINDOW_POINTS, len(seg) + 1):
            window_min = minutes.iloc[i-SLOPE_WINDOW_POINTS:i]
            window_temp = seg['temp_de'].iloc[i-SLOPE_WINDOW_POINTS:i]
            if window_min.max() - window_min.min() > 0:
                slope = stats.linregress(window_min, window_temp).slope
                rolling_slopes.append((seg['timestamp'].iloc[i-1], slope))
        
        # 找第一个斜率 >= 急剧恶化阈值的时刻
        for ts, slope in rolling_slopes:
            if slope >= TEMP_SLOPE_SHARP:
                result["ramp_start_time"] = ts.isoformat()
                break
    
    return result


def _format_phases_for_llm(phases: list[dict]) -> str:
    """把分段数据格式化成 LLM 易读的文本块。"""
    if not phases:
        return "（无有效阶段数据）"
    lines = []
    for ph in phases:
        lines.append(
            f"◆ 阶段 [{ph['state']}] {ph['start']} ~ {ph['end']}\n"
            f"    时长 {ph['duration_sec']}s，{ph['data_points']} 点\n"
            f"    流量: avg={ph['avg_flow']} m³/h, CV={ph['cv_flow']}%\n"
            f"    出口压力: avg={ph['avg_press_out']} MPa, CV={ph['cv_press']}%\n"
            f"    驱动端温度: slope={ph['slope_temp_de']} ℃/min, max={ph['max_temp_de']} ℃\n"
            f"    驱动端振动: slope={ph['slope_vib_de']} mm/s/min, max={ph['max_vib_de']} mm/s\n"
            f"    非驱动端温度: max={ph['max_temp_nde']} ℃, 振动: max={ph['max_vib_nde']} mm/s\n"
            f"    电机电流: avg={ph['avg_motor_current']} A, max={ph['max_motor_current']} A"
        )
    return "\n\n".join(lines)


def _format_overall(overall: dict) -> str:
    """把 overall dict 格式化成 LLM 易读的文本块。"""
    if not overall:
        return "（无全局概要）"
    return (
        f"- 总数据点数：{overall.get('total_points', 0)}\n"
        f"- 时间窗口：{overall.get('window_start', '?')} ~ {overall.get('window_end', '?')}\n"
        f"- 起始状态：{overall.get('start_state', 'UNKNOWN')}\n"
        f"- 结束状态：{overall.get('end_state', 'UNKNOWN')}\n"
        f"- 是否发生停机：{overall.get('has_shutdown', False)}\n"
        f"- 阶段数：{overall.get('phase_count', 0)}\n"
        f"- 全程最高温度(驱动端)：{overall.get('max_temp_de_overall', '?')} ℃\n"
        f"- 全程最高振动(驱动端)：{overall.get('max_vib_de_overall', '?')} mm/s"
    )

def _format_alarm_events_for_llm(events: list[dict]) -> str:
    """把报警段序列渲染成给大模型看的文本。

    输入是 query_alarm_events 的游程编码结果：**每段 = 一次连续报警**，
    带报警码、首末时间、持续时长，以及报警开始/结束时的设备状态与段内峰值。

    ★ 与旧版的区别：旧版是"每帧一行"，要靠 is_new_alarm 过滤后才勉强可用；
      现在是"每次报警一段"，天然无冗余帧，且多了"持续了多久"这个关键信息。
    """
    if not events:
        return "报警事件序列：（窗口内无报警事件）"

    lines = ["报警事件序列（每段 = 一次连续报警，含持续时长与报警时的设备状态）："]
    for ev in events:
        # 被上限保护折叠的摘要行
        if "note" in ev:
            lines.append(f"  [{ev['alarm_code']}] {ev['note']}")
            continue

        s, e, pk = ev["state_at_start"], ev["state_at_end"], ev["peak"]
        lines.append(
            f"  {ev['alarm_code']} | {ev['start'][-8:]}~{ev['end'][-8:]}"
            f" | 持续{ev['duration_sec']}s | {ev['start_state']}→{ev['end_state']}"
            f" | 温度(de){s['temp_de']}→{e['temp_de']}℃(峰{pk['temp_de']})"
            f" | 振动(de){s['vib_rms_de']}→{e['vib_rms_de']}mm/s(峰{pk['vib_rms_de']})"
            f" | 流量{s['flow_rate']}m³/h"
        )
    return "\n".join(lines)