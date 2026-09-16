# =============================================================================
# Step 2 的取数函数
#
# 由节点（data_nodes.py）直接调用，**不是绑定给大模型的工具**，因此不加 @tool。
# 表结构与测点白名单在 repository.py，本文件只负责"怎么查"。
# =============================================================================
from datetime import datetime
from src.utils.database import get_db_session

from .repository import ScadaTelemetry

#: 记录"报警时设备状态"用的 8 个测点。
#: 从表结构自动推导，避免手写白名单与 ORM 定义漂移（加测点时不会漏改）。
_NON_METRIC_COLUMNS = {"id", "timestamp", "device_id", "operating_state", "alarm_code", "created_at"}
STATE_FIELDS: tuple[str, ...] = tuple(
    c.name for c in ScadaTelemetry.__table__.columns if c.name not in _NON_METRIC_COLUMNS
)

#: 报警段数上限。超过后按持续时长保留最长的若干段，其余折叠成一条摘要。
#: 这是**数据整形**的上限，不是诊断阈值，所以放在这里而不是 rules/thresholds.py。
MAX_ALARM_RUNS = 30

#: 段间断容忍（秒）。相邻两次同码报警若间隔不超过此值，视为**同一次报警**。
#:
#: 为什么需要去抖：报警码是按"阈值 + 持续时长"判出来的，测点在阈值附近抖动时
#: 会出现 `TAH-101 → NONE → TAH-101` 这样的闪烁（实测数据里 08:00:05 报了一次、
#: 08:00:10 掉、08:00:15 又报）。不去抖的话，一次持续 2 分半的报警会被切成
#: "1 帧的段 + 145 秒的段"两条，既看不出真实持续时间，也白白多占一条记录。
#: 默认 10s = 2 个采样周期。
ALARM_GAP_TOLERANCE_SEC = 10


def _parse_ts(ts: str) -> datetime:
    """把 Step 1 传来的 ISO 字符串解析为 naive datetime。

    统一在工具入口把 ISO 字符串 parse 成 naive datetime，
    避免 MySQL 隐式转换在遇到 'Z' 后缀时行为不一致。
    """
    return datetime.fromisoformat(ts.replace("Z", ""))


def query_scada_telemetry(device_id: str, start_time: str, end_time: str) -> list:
    """按设备 + 时间窗口（闭区间）查询 SCADA 时序数据，返回 dict 列表供 Pandas 消费。"""

    start_dt, end_dt = _parse_ts(start_time), _parse_ts(end_time)

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


def _metrics_of(row) -> dict:
    """取出该帧的 8 个测点值（用于记录"报警时设备处于什么状态"）。"""
    return {f: getattr(row, f) for f in STATE_FIELDS}


def _summarize_run(code: str, rows: list) -> dict:
    """把一次连续报警的所有帧，压缩成一条"报警段"记录。

    ★ 这就是游程编码：5s 采样下同一报警码会连续出现几十上百帧，
      但真正有信息量的是"这条报警从什么时候开始、持续多久、当时设备什么状态"。
      压缩后条数降 6~12 倍，信息反而更全（多了持续时长与首末状态）。
    """
    first, last = rows[0], rows[-1]
    return {
        "alarm_code": code,
        "start": first.timestamp.isoformat(),
        "end": last.timestamp.isoformat(),
        "duration_sec": int((last.timestamp - first.timestamp).total_seconds()),
        "data_points": len(rows),
        "start_state": first.operating_state.value if first.operating_state else "UNKNOWN",
        "end_state": last.operating_state.value if last.operating_state else "UNKNOWN",
        # 报警开始 / 结束时刻的设备状态
        "state_at_start": _metrics_of(first),
        "state_at_end": _metrics_of(last),
        # 该段内的测点峰值
        "peak": {f: max(getattr(r, f) for r in rows) for f in STATE_FIELDS},
    }


def _cap_runs(runs: list, limit: int = MAX_ALARM_RUNS) -> list:
    """报警段数上限保护。

    为什么需要：报警码在阈值附近抖动时（碎片场景）会产生几十上百个极短段，
    若不限制会把 state 和提示词一起撑爆。
    保留策略：按**持续时长降序**取前 limit-1 段 —— 抖动产生的短段信息量最低，
    真正要看的是持续存在的报警；其余折叠成一条摘要，不静默丢弃。
    """
    if len(runs) <= limit:
        return runs

    ordered = sorted(runs, key=lambda x: x["duration_sec"], reverse=True)
    keep, dropped = ordered[: limit - 1], ordered[limit - 1:]
    keep.sort(key=lambda x: x["start"])          # 恢复时间顺序
    total_sec = sum(d["duration_sec"] for d in dropped)
    keep.append({
        "alarm_code": "（折叠）",
        "start": dropped[0]["start"],
        "end": dropped[-1]["end"],
        "note": (
            f"另有 {len(dropped)} 段短促报警未逐条列出"
            + (f"，合计 {total_sec} 秒" if total_sec else "（均为单帧触发）")
        ),
    })
    return keep


def query_alarm_events(device_id: str, start_time: str, end_time: str) -> list:
    """查询窗口内的**报警段**（游程编码），用于查看报警发生时设备的状态。

    与 query_scada_telemetry 的区别：
      前者返回逐帧遥测；本函数返回"一次连续报警 = 一条记录"，
      每条包含：报警码、首末时间、持续秒数、报警开始/结束时的设备状态、段内峰值。

    ★ 为什么要首末时间：单看"某时刻报了什么码"信息量很低，
      而"这个码从 08:00:05 持续到 08:02:40、期间温度从 70.5 升到 74.2"
      才回答了"这次报警到底意味着什么"。

    ★ 去抖：相邻两次同码报警若间隔 ≤ ALARM_GAP_TOLERANCE_SEC，合并为同一次。
      否则阈值的边界抖动会把一次报警切成好几段（实测把 2 分半切成了 1 帧 + 145 秒）。
    """
    start_dt, end_dt = _parse_ts(start_time), _parse_ts(end_time)

    #: 已聚合的原始段：{"code": str, "rows": [...]}
    raw: list[dict] = []
    #: 上一个"有报警"帧的时间戳 —— 用于算段间断间隔（跳过 NONE 帧）
    last_alarm_ts = None

    with get_db_session() as db:
        rows = db.query(ScadaTelemetry).filter(
            ScadaTelemetry.device_id == device_id,
            ScadaTelemetry.timestamp >= start_dt,
            ScadaTelemetry.timestamp <= end_dt,
        ).order_by(ScadaTelemetry.timestamp.asc()).all()

        for r in rows:
            code = (r.alarm_code or "NONE").strip()
            if code in ("", "NONE", "None"):
                continue                      # NONE 帧本身不入段，只影响间隔计算

            gap = (
                (r.timestamp - last_alarm_ts).total_seconds()
                if last_alarm_ts is not None else 0.0
            )
            same_run = (
                raw
                and raw[-1]["code"] == code
                and gap <= ALARM_GAP_TOLERANCE_SEC
            )
            if same_run:
                raw[-1]["rows"].append(r)     # 去抖：并入上一段
            else:
                raw.append({"code": code, "rows": [r]})
            last_alarm_ts = r.timestamp

        # ★ 必须在 with 块内完成汇总：Session 关闭后 ORM 实例会 detached，
        #   再去读 r.timestamp 会抛 DetachedInstanceError。
        runs = [_summarize_run(x["code"], x["rows"]) for x in raw]

    return _cap_runs(runs)