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
    RULE_CATALOG,
    SLOPE_WINDOW_POINTS,
    TEMP_SLOPE_ACTIVATION_C as _TEMP_SLOPE_ACTIVATION,
    TEMP_SLOPE_MIN_DURATION_SEC as _TEMP_SLOPE_MIN_DURATION_SEC,
    TEMP_SLOPE_MIN_POINTS as _TEMP_SLOPE_MIN_POINTS,
    TEMP_SLOPE_SHARP,
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


def _extract_alarm_codes(df: pd.DataFrame) -> tuple[str, str, list[str]]:
    """从窗口内的帧里提取报警码，返回三元组 ``(effective, last, all)``。

    参数：
        df: 原始遥测 DataFrame，必须含 ``alarm_code`` 列
            （多码以分号连接，无报警为 ``NONE``）。

    返回：
        ``effective``: 窗口内出现过的报警码去重排序（";" 连接）；无则 ``"NONE"``
        ``last``:      窗口内**最后一次**非 NONE 的报警码组合；无则 ``"NONE"``
        ``all``:       去重后的列表形式；无则 ``[]``
    """
    valid_rows = df[df['alarm_code'].apply(_is_valid_alarm)]
    if valid_rows.empty:
        return "NONE", "NONE", []

    # 收集窗口内出现过的所有报警码
    all_codes = set()
    for codes_str in valid_rows['alarm_code']:
        for code in codes_str.split(";"):
            code = code.strip()
            if code:
                all_codes.add(code)
    all_list = sorted(all_codes)

    effective = ";".join(all_list) if all_list else "NONE"
    last = valid_rows['alarm_code'].iloc[-1].strip()
    return effective, last, all_list


def _detect_ramp_start(seg: pd.DataFrame) -> dict:
    """检测段内温度"开始急剧恶化"的时刻。

    参数：
        seg: 单个状态段的 DataFrame，需含 ``timestamp``（已是 datetime）与 ``temp_de``。

    返回：
        ``{"ramp_start_time": <ISO 时刻字符串 或 None>}`` ——
        段内温度用 1 分钟滑窗做线性回归，斜率**首次**达到
        ``TEMP_SLOPE_SHARP``（0.8 ℃/min）的那一帧时刻。
        段长不足 1 分钟、或全程没到过该斜率时为 None。

    用途：
        让 ``_judge_flags`` 能在告警文案里写"约从 08:01:00 开始"，
        而不是只说"这一段斜率很高" —— 定位到时刻才对处置有用。
    """
    result = {"ramp_start_time": None}

    if len(seg) < SLOPE_WINDOW_POINTS:  # 至少 1 分钟数据才谈得上"滑窗斜率"
        return result

    t0 = seg['timestamp'].min()
    minutes = (seg['timestamp'] - t0).dt.total_seconds() / 60

    # 滑动窗口回归（每 1 分钟 = SLOPE_WINDOW_POINTS 点），找第一个越阈值的时刻
    for i in range(SLOPE_WINDOW_POINTS, len(seg) + 1):
        window_min = minutes.iloc[i-SLOPE_WINDOW_POINTS:i]
        window_temp = seg['temp_de'].iloc[i-SLOPE_WINDOW_POINTS:i]
        if window_min.max() - window_min.min() <= 0:
            continue
        if stats.linregress(window_min, window_temp).slope >= TEMP_SLOPE_SHARP:
            result["ramp_start_time"] = seg['timestamp'].iloc[i-1].isoformat()
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


def render_rule_hits(phases: list[dict]) -> str:
    """把分段上的规则码渲染成大模型可读的中文事实句（**只在提示词里用，不进 state**）。

    参数：
        phases: 含 ``rule_hits`` 的分段列表（即 ``calculated_metrics["phases"]``）。

    返回：
        每行一条，形如
        ``- [WARNING段 08:00:05~08:09:55] 驱动端温度超停机线（BEARING_TEMP_DE_TRIP）``；
        一条命中都没有时返回 ``"无"``。
    """
    lines = []
    for ph in phases:
        for code in ph.get("rule_hits", []):
            spec = RULE_CATALOG.get(code, {})
            label = spec.get("label", code)

            extra = ""
            extra_field = spec.get("extra_field")
            if extra_field and ph.get(extra_field):
                extra = spec["extra_template"].format(v=str(ph[extra_field])[-8:])

            lines.append(
                f"- [{ph['state']}段 {ph['start'][-8:]}~{ph['end'][-8:]}] "
                f"{label}（{code}）{extra}"
            )
    return "\n".join(lines) or "无"


def _format_overall(overall: dict) -> str:
    """把 overall dict 格式化成 LLM 易读的文本块。

    参数：
        overall: Node ``analyze`` 产出的窗口概要（``calculated_metrics.overall``）。

    返回：
        多行文本；``overall`` 为空时返回"（无全局概要）"。
    """
    if not overall:
        return "（无全局概要）"
    return (
        f"- 总数据点数：{overall.get('total_points', 0)}\n"
        f"- 时间窗口：{overall.get('window_start', '?')} ~ {overall.get('window_end', '?')}\n"
        f"- 起始状态：{overall.get('start_state', 'UNKNOWN')}\n"
        f"- 结束状态：{overall.get('end_state', 'UNKNOWN')}\n"
        f"- 是否发生停机：{overall.get('has_shutdown', False)}\n"
        f"- 阶段数：{overall.get('phase_count', 0)}\n"
        f"- 窗口内出现过的报警码：{overall.get('effective_alarm_codes', 'NONE')}\n"
        f"- 全程最高温度(驱动端)：{overall.get('max_temp_de_overall', '?')} ℃\n"
        f"- 全程最高振动(驱动端)：{overall.get('max_vib_de_overall', '?')} mm/s"
    )
