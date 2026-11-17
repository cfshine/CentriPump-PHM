# =============================================================================
# Step 2 的取数函数
# 表结构与测点白名单在 repository.py，本文件只负责"怎么查"。
# =============================================================================
from datetime import datetime


def _parse_ts(ts: str) -> datetime:
    """把 Step 1 传来的 ISO 字符串解析为 naive datetime。

    参数：
        ts: ISO 8601 时间字符串，允许带 'Z' 后缀（如 ``2026-09-13T08:00:00Z``）。

    返回：
        naive ``datetime``（去掉时区信息）。
    """
    return datetime.fromisoformat(ts.replace("Z", ""))


def query_scada_telemetry(device_id: str, start_time: str, end_time: str, *, csv_path: str = "") -> list:
    """按设备 + 时间窗口（闭区间）查询 SCADA 时序数据。

    参数：
        device_id:  设备资产位号（如 ``PUMP-IS100-80-160-01``）。
        start_time: 窗口起点，ISO 字符串。
        end_time:   窗口终点，ISO 字符串。

    返回：
        list[dict]，每帧一条，含 timestamp / 9 个测点 / operating_state / alarm_code。
        窗口内无数据时返回空列表（不抛异常）。
    """
    start_dt, end_dt = _parse_ts(start_time), _parse_ts(end_time)

    if csv_path:
        import numpy as np
        import pandas as pd

        df = pd.read_csv(csv_path, keep_default_na=False)
        required = {"device_id", "timestamp", "flow_rate", "press_out", "press_in",
                    "temp_de", "temp_nde", "vib_rms_de", "vib_rms_nde",
                    "motor_current", "operating_state", "alarm_code"}
        if missing := required - set(df.columns):
            raise ValueError(f"CSV 缺少字段: {sorted(missing)}")
        times = pd.to_datetime(df["timestamp"], errors="raise")
        df = df.loc[(df.device_id == device_id) & times.between(start_dt, end_dt)].copy()
        numeric = sorted(required - {"device_id", "timestamp", "operating_state", "alarm_code"})
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
            raise ValueError("CSV 测点存在空值或非有限数值")
        if df["timestamp"].duplicated().any():
            raise ValueError("CSV 同设备时间戳重复")
        return df.sort_values("timestamp").to_dict("records")

    from src.utils.database import get_db_session
    from .repository import ScadaTelemetry

    with get_db_session() as db:
        records = db.query(ScadaTelemetry).filter(
            ScadaTelemetry.device_id == device_id,
            ScadaTelemetry.timestamp >= start_dt,
            ScadaTelemetry.timestamp <= end_dt,
        ).order_by(ScadaTelemetry.timestamp.asc()).all()

        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "flow_rate": r.flow_rate,
                "press_out": r.press_out,
                "press_in": r.press_in,
                "temp_de": r.temp_de,
                "temp_nde": r.temp_nde,
                "vib_rms_de": r.vib_rms_de,
                "vib_rms_nde": r.vib_rms_nde,
                "motor_current": r.motor_current,
                "operating_state": r.operating_state.value,
                "alarm_code": r.alarm_code,
            }
            for r in records
        ]
