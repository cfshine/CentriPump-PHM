#utils/math.py
import pandas as pd
import numpy as np
# ★ 全部数值判据来自 rules/thresholds.py（唯一出处），本文件不再出现魔法数字
from rules.thresholds import (
    CURRENT_HIGH_A,
    CURRENT_LOW_A,
    CV_CAVITATION_PCT,
    CV_MIN_POINTS,
    CV_WINDOW_SEC,
    DEV_FLOW_PCT,
    DEV_PRESS_PCT,
    PRESS_IN_WARN_MPA,
    RATED_FLOW_M3H,
    RATED_PRESS_OUT_MPA,
    TEMP_SLOPE_SHARP,
    TEMP_SLOPE_SLOW,
    TEMP_TRIP_C,
    TEMP_WARN_C,
    VIB_SYNC_HIGH_MM_S,
    VIB_SYNC_LOW_MM_S,
    VIB_TRIP_MM_S,
    VIB_WARN_MM_S,
)
from src.sub_agents.data_agent.timeseries_tools import (
    _py, 
    _cv, 
    _ge,
    _gt,
    _lt,
    _safe_slope, 
    _detect_ramp_start,
    _ACTIVE_STATES, 
    _FLOW_ACTIVE_THRESHOLD, 
    _TEMP_SLOPE_ACTIVATION, 
    _TEMP_SLOPE_MIN_DURATION_SEC, 
    _TEMP_SLOPE_MIN_POINTS
    )


def _segment_metrics(seg: pd.DataFrame) -> dict:
    """计算单个阶段段的统计指标（全部转 Python 原生类型）。"""
    t0 = seg['timestamp'].min()
    t1 = seg['timestamp'].max()
    seg_minutes = (seg['timestamp'] - t0).dt.total_seconds() / 60

    return {
        "start": t0.isoformat(),
        "end": t1.isoformat(),
        "duration_sec": int((t1 - t0).total_seconds()),
        "data_points": int(len(seg)),

        # 水力
        "avg_flow": _py(seg['flow_rate'].mean(), 2),
        "avg_press_out": _py(seg['press_out'].mean(), 3),
        "avg_press_in": _py(seg['press_in'].mean(), 3),
        "cv_flow": _py(_cv(seg['flow_rate']), 2),
        "cv_press": _py(_cv(seg['press_out']), 2),

        # 温度
        # ★ 只给驱动端算斜率：非驱动端斜率（原 slope_temp_nde）全项目没有任何地方读，
        #   2026-09-17 删除。非驱动端仍保留峰值 max_temp_nde（叙事要用）。
        "slope_temp_de": _py(_safe_slope(seg_minutes, seg['temp_de']), 3),
        "max_temp_de": _py(seg['temp_de'].max(), 1),
        "max_temp_nde": _py(seg['temp_nde'].max(), 1),

        # 振动（同上：只给驱动端算斜率）
        "slope_vib_de": _py(_safe_slope(seg_minutes, seg['vib_rms_de']), 3),
        "max_vib_de": _py(seg['vib_rms_de'].max(), 2),
        "max_vib_nde": _py(seg['vib_rms_nde'].max(), 2),

        # 电气
        "avg_motor_current": _py(seg['motor_current'].mean(), 2),
        "max_motor_current": _py(seg['motor_current'].max(), 2),

        # 滑窗 CV 峰值
        "cv_flow_peak": _py(_rolling_cv_peak(seg['flow_rate'], seg['timestamp']), 2),
        "cv_press_peak": _py(_rolling_cv_peak(seg['press_out'], seg['timestamp']), 2),

        **_detect_ramp_start(seg),
    }


def _judge_flags(phases: list[dict]) -> list[str]:
    """逐段做确定性规则判定，把**机器码**写进每段的 ``rule_hits``，并返回窗口级码列表。

    参数：
        phases: ``_segment_metrics`` 产出的分段列表。
                本函数会**原地**给每个 phase 写入 ``rule_hits``
                （该段命中的规则码列表；停机段写入空列表）。

    返回：
        list[str]：窗口内命中过的规则码，按**首次触发顺序**去重。

    为什么码要落到"每一段"上，而不是只给窗口一个扁平列表：
        · 拼提示词时不必反推"哪一段触发了哪条码" —— 反推会与真实判定漂移
          （温度斜率类判据还带 ≥60℃、段长 ≥180s、点数 ≥36 三道上下文门槛）；
        · 下游能看到"哪一段出了什么问题"，比一个扁平列表信息量大。

    为什么不再拼中文句子：
        中文事实句改由 ``render_rule_hits`` 在拼提示词时生成（只活在 prompt 里，
        不进 state）。state 只留机器码：短、稳定、可溯源，文案改字不影响下游。

    ★ 判据本身一个字都没改，阈值仍全部来自 rules/thresholds.py（唯一出处）。
    """
    window_hits: list[str] = []

    for ph in phases:
        # 停机是故障的**结果**：其低流量/低振动/低电流不是异常源，照常判定会刷一堆假告警
        if ph["state"] not in _ACTIVE_STATES:
            ph["rule_hits"] = []
            continue

        hits: list[str] = []          # 本段命中的规则码（原先是中文句子）

        # ========== 温度判据 ==========
        st = ph["slope_temp_de"]
        max_temp = ph["max_temp_de"]

        # 温度绝对值判据（无条件触发）
        if _ge(max_temp, TEMP_TRIP_C):
            hits.append("BEARING_TEMP_DE_TRIP")
        elif _ge(max_temp, TEMP_WARN_C):
            hits.append("BEARING_TEMP_DE_WARN")

        # 温度斜率判据（带上下文约束）
        slope_judge_enabled = (
            _ge(max_temp, TEMP_TRIP_C)
            or (
                _ge(max_temp, _TEMP_SLOPE_ACTIVATION)
                and ph["duration_sec"] >= _TEMP_SLOPE_MIN_DURATION_SEC
                and ph["data_points"] >= _TEMP_SLOPE_MIN_POINTS
            )
        )

        if slope_judge_enabled and st is not None:
            # "急剧恶化" 必须同时满足：
            #   1) 斜率 ≥ 0.8 ℃/min
            #   2) 段内温度至少跨过 70℃ 预警线
            # 否则高斜率但低温（如 DEGRADING 段 45→65℃）只能算"缓慢劣化"，
            # 避免把正常劣化段误判为紧急恶化。
            # （"约从 08:01:00 开始"这句上下文由渲染器从 ramp_start_time 补上）
            if st >= TEMP_SLOPE_SHARP and _ge(max_temp, TEMP_WARN_C):
                hits.append("BEARING_TEMP_DE_RAMP_SHARP")
            elif st >= TEMP_SLOPE_SLOW:
                hits.append("BEARING_TEMP_DE_RAMP_SLOW")

        # ========== 振动判据 ==========
        # 驱动端单端
        if _gt(ph["max_vib_de"], VIB_TRIP_MM_S):
            hits.append("VIB_DE_TRIP")
        elif _gt(ph["max_vib_de"], VIB_WARN_MM_S):
            hits.append("VIB_DE_WARN")

        # 非驱动端单端
        if _gt(ph["max_vib_nde"], VIB_TRIP_MM_S):
            hits.append("VIB_NDE_TRIP")
        elif _gt(ph["max_vib_nde"], VIB_WARN_MM_S):
            hits.append("VIB_NDE_WARN")

        # 两端差异判据，中性描述，不做故障归因
        vib_de = ph["max_vib_de"] or 0.0
        vib_nde = ph["max_vib_nde"] or 0.0
        if vib_de > VIB_SYNC_HIGH_MM_S and vib_nde > VIB_SYNC_HIGH_MM_S:
            hits.append("VIB_BOTH_HIGH")
        elif vib_de > VIB_SYNC_HIGH_MM_S and vib_nde < VIB_SYNC_LOW_MM_S:
            hits.append("VIB_DE_DOMINANT")
        elif vib_nde > VIB_SYNC_HIGH_MM_S and vib_de < VIB_SYNC_LOW_MM_S:
            hits.append("VIB_NDE_DOMINANT")

        # ========== 水力判据（流量 / 出口压力偏离额定值） ==========
        # 两条判据共用同一道**只看物理量**的闸门：流量 < 20 m³/h 视为停机或尚未
        # 建立流动，此时流量低、压力低都是必然结果，判"偏离额定值"只会产生假告警
        # （实测：停机后刚重启的那几帧 avg_flow=0.92、压头偏离 -98%，被这道闸门挡住）。
        if _gt(ph["avg_flow"], _FLOW_ACTIVE_THRESHOLD):
            dev_flow = (ph["avg_flow"] - RATED_FLOW_M3H) / RATED_FLOW_M3H * 100
            if abs(dev_flow) > DEV_FLOW_PCT:
                hits.append("FLOW_DEV")

            # 出口压力偏离额定值：判"泵没把额定扬程打出来"（规范第五节 5.1）。
            # 实测唯一触发点是重启欠载段（流量 42 m³/h、出口压力只有额定的 42%）。
            dev_press = (ph["avg_press_out"] - RATED_PRESS_OUT_MPA) / RATED_PRESS_OUT_MPA * 100
            if abs(dev_press) > DEV_PRESS_PCT:
                hits.append("PRESS_OUT_DEV")

        # ========== 气蚀判据（滑窗 CV 优先） ==========
        if ph["data_points"] >= CV_MIN_POINTS:
            cv_flow_peak = ph.get("cv_flow_peak")
            if cv_flow_peak is None:
                cv_flow_peak = ph.get("cv_flow") or 0.0
            cv_press_peak = ph.get("cv_press_peak")
            if cv_press_peak is None:
                cv_press_peak = ph.get("cv_press") or 0.0

            if _ge(cv_flow_peak, CV_CAVITATION_PCT) or _ge(cv_press_peak, CV_CAVITATION_PCT):
                hits.append("CV_UNSTABLE")

        # ========== 入口压力判据 ==========
        if _lt(ph["avg_press_in"], PRESS_IN_WARN_MPA):
            hits.append("PRESS_IN_LOW")

        # ========== 电机电流判据 ==========
        if _ge(ph["max_motor_current"], CURRENT_HIGH_A):
            hits.append("CURRENT_HIGH")
        elif _lt(ph["avg_motor_current"], CURRENT_LOW_A):
            hits.append("CURRENT_LOW")

        # —— 本段落盘：段级码 + 并入窗口级去重列表 ——
        ph["rule_hits"] = hits
        for code in hits:
            if code not in window_hits:
                window_hits.append(code)

    return window_hits

def _rolling_cv_peak(series: pd.Series, timestamps: pd.Series,
                     window_sec: int = CV_WINDOW_SEC, min_points: int = CV_MIN_POINTS) -> float:
    """
    在时间窗口内滑动计算 CV，返回滑窗 CV 的最大值。
    用来捕获"局部剧烈波动"（如气蚀段），避免段级 CV 被平稳期拉平。
    """
    if len(series) < min_points:
        return 0.0

    points_per_window = max(min_points, window_sec // 5)
    if points_per_window > len(series):
        points_per_window = len(series)

    rolling_mean = series.rolling(window=points_per_window, min_periods=min_points).mean()
    rolling_std = series.rolling(window=points_per_window, min_periods=min_points).std()

    with np.errstate(divide='ignore', invalid='ignore'):
        rolling_cv = pd.Series(
            np.where(
                (rolling_mean > 1e-6) & rolling_std.notna(),
                rolling_std / rolling_mean * 100,
                0.0,
            ),
            index=series.index,
        )

    # 显式过滤 NaN
    rolling_cv = rolling_cv.dropna()
    if rolling_cv.empty:
        return 0.0

    peak = float(rolling_cv.max())
    return 0.0 if pd.isna(peak) else round(peak, 2)
