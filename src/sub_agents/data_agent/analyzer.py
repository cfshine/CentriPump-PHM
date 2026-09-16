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
    PRESS_IN_WARN_MPA,
    RATED_FLOW_M3H,
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
    _detect_inflection,
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
        "slope_temp_de": _py(_safe_slope(seg_minutes, seg['temp_de']), 3),
        "slope_temp_nde": _py(_safe_slope(seg_minutes, seg['temp_nde']), 3),
        "max_temp_de": _py(seg['temp_de'].max(), 1),
        "max_temp_nde": _py(seg['temp_nde'].max(), 1),

        # 振动
        "slope_vib_de": _py(_safe_slope(seg_minutes, seg['vib_rms_de']), 3),
        "slope_vib_nde": _py(_safe_slope(seg_minutes, seg['vib_rms_nde']), 3),
        "max_vib_de": _py(seg['vib_rms_de'].max(), 2),
        "max_vib_nde": _py(seg['vib_rms_nde'].max(), 2),

        # 电气
        "avg_motor_current": _py(seg['motor_current'].mean(), 2),
        "max_motor_current": _py(seg['motor_current'].max(), 2),

        # 滑窗 CV 峰值
        "cv_flow_peak": _py(_rolling_cv_peak(seg['flow_rate'], seg['timestamp']), 2),
        "cv_press_peak": _py(_rolling_cv_peak(seg['press_out'], seg['timestamp']), 2),

        **_detect_inflection(seg),
    }

def _judge_flags(phases: list[dict]) -> list[str]:

    """根据阶段段统计指标，生成报警/预警标签。"""

    flags = []
    for ph in phases:
        if ph["state"] not in _ACTIVE_STATES:
            continue
        tag = f"[{ph['state']}段]"

        # ========== 温度判据 ==========
        st = ph["slope_temp_de"]
        max_temp = ph["max_temp_de"]

        # 温度绝对值判据（无条件触发）
        if _ge(max_temp, TEMP_TRIP_C):
            flags.append(f"{tag} 驱动端温度超停机线 (max {max_temp:.1f} ≥ 80℃)")
        elif _ge(max_temp, TEMP_WARN_C):
            flags.append(f"{tag} 驱动端温度超预警线 (max {max_temp:.1f} ≥ 70℃)")

        # 温度斜率判据（带上下文约束 + 拐点信息）
        slope_judge_enabled = (
            _ge(max_temp, TEMP_TRIP_C)
            or (
                _ge(max_temp, _TEMP_SLOPE_ACTIVATION)
                and ph["duration_sec"] >= _TEMP_SLOPE_MIN_DURATION_SEC
                and ph["data_points"] >= _TEMP_SLOPE_MIN_POINTS
            )
        )

        if slope_judge_enabled and st is not None:
            ramp_info = ""
            if ph.get("ramp_start_time"):
                ramp_ts = ph["ramp_start_time"][-8:]
                ramp_info = f"，约从 {ramp_ts} 开始"
            
            # "急剧恶化" 必须同时满足：
            #   1) 斜率 ≥ 0.8 ℃/min
            #   2) 段内温度至少跨过 70℃ 预警线
            # 否则高斜率但低温（如 DEGRADING 段 45→65℃）只能算"缓慢劣化"，
            # 避免把正常劣化段误判为紧急恶化。
            if st >= TEMP_SLOPE_SHARP and _ge(max_temp, TEMP_WARN_C):
                flags.append(
                    f"{tag} 驱动端温度急剧恶化 "
                    f"(斜率 {st:.2f} ℃/min ≥ {TEMP_SLOPE_SHARP}，红色紧急{ramp_info})"
                )
            elif st >= TEMP_SLOPE_SLOW:
                flags.append(
                    f"{tag} 驱动端温度缓慢劣化 "
                    f"(斜率 {st:.2f} ℃/min，黄色关注{ramp_info})"
                )

        # ========== 振动判据 ==========
        # 驱动端单端
        if _gt(ph["max_vib_de"], VIB_TRIP_MM_S):
            flags.append(f"{tag} 驱动端振动超国标停机线 (max {ph['max_vib_de']:.2f} > {VIB_TRIP_MM_S} mm/s)")
        elif _gt(ph["max_vib_de"], VIB_WARN_MM_S):
            flags.append(f"{tag} 驱动端振动超良好区上限 (max {ph['max_vib_de']:.2f} > {VIB_WARN_MM_S} mm/s)")

        # 非驱动端单端
        if _gt(ph["max_vib_nde"], VIB_TRIP_MM_S):
            flags.append(f"{tag} 非驱动端振动超国标停机线 (max {ph['max_vib_nde']:.2f} > {VIB_TRIP_MM_S} mm/s)")
        elif _gt(ph["max_vib_nde"], VIB_WARN_MM_S):
            flags.append(f"{tag} 非驱动端振动超良好区上限 (max {ph['max_vib_nde']:.2f} > {VIB_WARN_MM_S} mm/s)")

        # 两端差异判据，中性描述，不做故障归因
        vib_de = ph["max_vib_de"] or 0.0
        vib_nde = ph["max_vib_nde"] or 0.0
        if vib_de > VIB_SYNC_HIGH_MM_S and vib_nde > VIB_SYNC_HIGH_MM_S:
            diff_pct = abs(vib_de - vib_nde) / max(vib_de, vib_nde) * 100
            flags.append(
                f"{tag} 两端振动同步偏高 "
                f"(de={vib_de:.2f}, nde={vib_nde:.2f}, 差值 {diff_pct:.1f}%)"
            )
        elif vib_de > VIB_SYNC_HIGH_MM_S and vib_nde < VIB_SYNC_LOW_MM_S:
            flags.append(
                f"{tag} 驱动端振动显著高于非驱动端 "
                f"(de={vib_de:.2f}, nde={vib_nde:.2f})"
            )
        elif vib_nde > VIB_SYNC_HIGH_MM_S and vib_de < VIB_SYNC_LOW_MM_S:
            flags.append(
                f"{tag} 非驱动端振动显著高于驱动端 "
                f"(de={vib_de:.2f}, nde={vib_nde:.2f})"
            )

        # ========== 流量判据 ==========
        if _gt(ph["avg_flow"], _FLOW_ACTIVE_THRESHOLD):
            dev_flow = (ph["avg_flow"] - RATED_FLOW_M3H) / RATED_FLOW_M3H * 100
            if abs(dev_flow) > DEV_FLOW_PCT:
                flags.append(f"{tag} 流量工况偏离额定值 >15% (实际 {dev_flow:+.1f}%)")

        # ========== 气蚀判据（滑窗 CV 优先） ==========
        cv_flow_peak = ph.get("cv_flow_peak") or ph.get("cv_flow") or 0.0
        cv_press_peak = ph.get("cv_press_peak") or ph.get("cv_press") or 0.0
        if ph["data_points"] >= CV_MIN_POINTS:
            cv_flow_peak = ph.get("cv_flow_peak")
            if cv_flow_peak is None:
                cv_flow_peak = ph.get("cv_flow") or 0.0
            cv_press_peak = ph.get("cv_press_peak")
            if cv_press_peak is None:
                cv_press_peak = ph.get("cv_press") or 0.0

            if _ge(cv_flow_peak, CV_CAVITATION_PCT) or _ge(cv_press_peak, CV_CAVITATION_PCT):
                flags.append(
                    f"{tag} 2分钟滑窗内流量/压力高频波动 "
                    f"(CV_flow_peak={cv_flow_peak:.1f}%, "
                    f"CV_press_peak={cv_press_peak:.1f}%，疑似流态失稳)"
                )

        # ========== 入口压力判据 ==========
        if _lt(ph["avg_press_in"], PRESS_IN_WARN_MPA):
            flags.append(f"{tag} 入口压力低 (avg {ph['avg_press_in']:.3f} < {PRESS_IN_WARN_MPA} MPa)")

        # ========== 电机电流判据 ==========
        if _ge(ph["max_motor_current"], CURRENT_HIGH_A):
            flags.append(f"{tag} 电机电流超额定满载 (max {ph['max_motor_current']:.2f} ≥ 29 A)")
        elif _lt(ph["avg_motor_current"], CURRENT_LOW_A):
            flags.append(f"{tag} 电机电流欠载 (avg {ph['avg_motor_current']:.2f} < 16 A)")

    return flags

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