#scripts/scada_generator.py
"""
SCADA 时序数据生成器
=================================
关键修复：
  1. operating_state 改为由 alarm_code 反推，保证两者一致
     （Bug: 流量<85 但状态标 NORMAL）
  2. scenario_cavitation 加 WARNING 过渡段
     （Bug: 气蚀→NORMAL 无过渡，5 秒内振动从 5.6 掉到 1.5）
  3. 启动过渡末期的 NORMAL 自动被推导为 WARNING
     （Bug: 流量仍<85 但状态变 NORMAL）

用法：
    python -m scripts.scada_generator --scenario bearing_dry
    python -m scripts.scada_generator --scenario full_day --clean
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
from sqlalchemy import delete

from src.utils.database import session_creater
from src.sub_agents.data_agent.repository import OperatingState, ScadaTelemetry


# ==================== 常量 ====================

SAMPLE_INTERVAL_SEC = 5
DEFAULT_DEVICE_ID = "PUMP-IS100-80-160-01"
DEFAULT_START_TIME = "2026-09-13T08:00:00"

# 额定工作点
RATED_FLOW = 100.0
RATED_PRESS_OUT = 0.312
RATED_PRESS_IN = 0.020
RATED_TEMP = 45.0
RATED_VIB = 1.45
RATED_CURRENT = 22.5


# ==================== Phase 数据类 ====================

@dataclass
class Phase:
    """
    一个时序阶段。
    state 字段现在仅作为"意图提示"（用于 DEGRADING 这种 alarm_code 不一定报的场景）；
    实际落库的 operating_state 由 _resolve_state() 结合测点值决定。
    """
    state: str
    duration_sec: int

    flow_rate:     Tuple[float, float, float] = (100.0, 100.0, 0.5)
    press_out:     Tuple[float, float, float] = (0.312, 0.312, 0.002)
    press_in:      Tuple[float, float, float] = (0.020, 0.020, 0.002)
    temp_de:       Tuple[float, float, float] = (45.0, 45.0, 0.4)
    temp_nde:      Tuple[float, float, float] = (43.0, 43.0, 0.4)
    vib_rms_de:    Tuple[float, float, float] = (1.45, 1.45, 0.06)
    vib_rms_nde:   Tuple[float, float, float] = (1.35, 1.35, 0.06)
    motor_current: Tuple[float, float, float] = (22.5, 22.5, 0.3)


# ==================== 报警码自动判定 ====================

def compute_alarm_code(row: dict, is_trip: bool = False) -> str:
    """
    依据《规范文档》第六节阈值规则自动判定报警码。
    停机状态下返回 NONE。
    """
    if is_trip:
        return "NONE"

    codes: set[str] = set()

    # 温度
    if row["temp_de"] >= 80.0 or row["temp_nde"] >= 80.0:
        codes.add("TAHH-101")
    elif row["temp_de"] >= 70.0 or row["temp_nde"] >= 70.0:
        codes.add("TAH-101")

    # 振动
    if row["vib_rms_de"] > 4.5 or row["vib_rms_nde"] > 4.5:
        codes.add("VAHH-102")
    elif row["vib_rms_de"] > 3.5 or row["vib_rms_nde"] > 3.5:
        codes.add("VAH-102")

    # 入口压力
    if row["press_in"] < -0.040:
        codes.add("PAL-103")

    # 流量
    if row["flow_rate"] < 85.0:
        codes.add("FAL-104")

    # 电流
    if row["motor_current"] >= 29.0:
        codes.add("IAH-106")
    elif row["motor_current"] < 15.0:
        codes.add("IAL-105")

    return ";".join(sorted(codes)) if codes else "NONE"


# ==================== 状态反推（核心修复） ====================
def _resolve_state(computed_alarm: str, phase_intent: str,
                   prev_state: str = None,
                   prev_alarm: str = "NONE",
                   temp_de: float = None) -> str:
    """
    由 alarm_code 反推状态，带迟滞（Hysteresis）。
    
    迟滞规则：
      1. 报警码从"有"变"无"时，至少保持一拍，避免阈值抖动
      2. 【新增】DEGRADING 中途不允许回 NORMAL：
         若上一状态是 DEGRADING 且温度仍接近预警线（≥65℃），
         保持 DEGRADING，避免在温度单调上升时出现 NORMAL 回弹
    """
    # 规则 1：停机优先级最高
    if phase_intent == "TRIP_SHUTDOWN":
        return "TRIP_SHUTDOWN"
    
    # 规则 2：特殊故障状态保留
    if phase_intent in ("CRITICAL_CAVITATION", "UNBALANCE_MISALIGNMENT"):
        return phase_intent
    
    # 规则 3：有报警码 → 至少 WARNING
    if computed_alarm and computed_alarm != "NONE":
        return "WARNING"
    
    # 规则 4：【迟滞】报警码刚清空时，不允许立即回 NORMAL
    if prev_state == "WARNING" and prev_alarm and prev_alarm != "NONE":
        return "WARNING"   # 保留一拍，让下一帧再判断
    
    # 规则 5：无报警码
    if phase_intent == "DEGRADING":
        return "DEGRADING"
    
    # 规则 5b：【新增】劣化中途不允许回 NORMAL
    # 场景：上一状态是 DEGRADING，当前温度仍在 65~70℃（尚未触发 TAH-101），
    #       但物理上温度仍在上升，不应回 NORMAL
    if (prev_state == "DEGRADING" 
            and temp_de is not None 
            and temp_de >= 65.0):
        return "DEGRADING"
    
    return "NORMAL"


# ==================== 数据生成核心 ====================

def _interp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


def generate_phase_rows(phase: Phase, start_dt: datetime,last_state: str = None,last_alarm: str = "NONE") -> tuple[List[dict], str, str]:
    """
    生成一个 Phase 内的所有采样点。
    返回 (rows, last_state, last_alarm)，把末帧状态透传给下一个 Phase。
    """
    n_points = phase.duration_sec // SAMPLE_INTERVAL_SEC
    if n_points <= 0:
        return [], last_state, last_alarm

    is_trip = (phase.state == "TRIP_SHUTDOWN")
    rows: List[dict] = []

    # 承接上一个 Phase 的末帧状态
    prev_state = last_state
    prev_alarm = last_alarm

    for i in range(n_points):
        progress = 0.0 if n_points == 1 else i / (n_points - 1)
        ts = start_dt + timedelta(seconds=SAMPLE_INTERVAL_SEC * i)

        def sample(triple):
            s, e, noise = triple
            return _interp(s, e, progress) + np.random.normal(0.0, noise)

        row = {
            "timestamp": ts,
            "flow_rate":     round(max(0.0, sample(phase.flow_rate)), 2),
            "press_out":     round(max(0.0, sample(phase.press_out)), 3),
            "press_in":      round(sample(phase.press_in), 3),
            "temp_de":       round(sample(phase.temp_de), 1),
            "temp_nde":      round(sample(phase.temp_nde), 1),
            "vib_rms_de":    round(max(0.0, sample(phase.vib_rms_de)), 2),
            "vib_rms_nde":   round(max(0.0, sample(phase.vib_rms_nde)), 2),
            "motor_current": round(max(0.0, sample(phase.motor_current)), 2),
        }

        computed_alarm = compute_alarm_code(row, is_trip=is_trip)
        state = _resolve_state(
            computed_alarm, phase.state,
            prev_state=prev_state,
            prev_alarm=prev_alarm,
            temp_de=row["temp_de"],
        )

        if state == "TRIP_SHUTDOWN":
            computed_alarm = "NONE"

        row["operating_state"] = state
        row["alarm_code"] = computed_alarm

        prev_state = state
        prev_alarm = computed_alarm
        rows.append(row)

    # 【关键】把末帧状态返回给上层
    return rows, prev_state, prev_alarm


def generate_scenario(phases: List[Phase], start_dt: datetime) -> List[dict]:
    """按顺序生成一个场景的所有采样点。"""
    all_rows: List[dict] = []
    cursor = start_dt

    # 【关键】跨 Phase 透传末帧状态
    last_state = None
    last_alarm = "NONE"

    for ph in phases:
        rows, last_state, last_alarm = generate_phase_rows(
            ph, cursor,
            last_state=last_state,
            last_alarm=last_alarm,
        )
        all_rows.extend(rows)
        cursor += timedelta(seconds=ph.duration_sec)

    return all_rows


# ==================== 场景定义 ====================

def scenario_normal(duration_min: int = 30) -> List[Phase]:
    """纯正常工况。"""
    return [Phase(state="NORMAL", duration_sec=duration_min * 60)]

def scenario_seal_leak() -> List[Phase]:
    """
    机械密封泄漏失效场景（24 小时完整周期）。
    
    物理演变链：
      密封面磨损 → 内漏增加 → 泵送效率下降 → 流量/压力缓降
        → 润滑油被冲刷 → 轴承摩擦增大 → 温度爬升 + 振动上升
        → 热失控 → 温度突破 80℃ → TRIP 停机
    
    报警触发顺序（真实时间序）：
      t+3h45m   FAL-104（流量穿破 85）
      t+6h20m   TAH-101（温度穿破 70）
      t+6h35m   TAHH-101（温度穿破 80）
      t+6h40m   VAHH-102（振动穿破 4.5）
      t+6h50m   TRIP_SHUTDOWN
    
    故障占比：约 4h / 24h ≈ 17%（高于真实 MTBF，但保留完整演变过程）
    """
    return [
        # ─────── 阶段 1：正常运行 ───────
        Phase(state="NORMAL", duration_sec=21600),   # 6h

        # ─────── 阶段 2：密封早期磨损 ───────
        # 特征：轻微内漏，效率微降，未触发任何报警
        Phase(
            state="DEGRADING",
            duration_sec=3600,   # 1h
            flow_rate=(100.0, 97.0, 0.5),
            press_out=(0.312, 0.305, 0.003),
            press_in=(0.020, 0.018, 0.002),
            temp_de=(45.5, 46.5, 0.4),
            temp_nde=(43.0, 44.0, 0.4),
            vib_rms_de=(1.50, 1.75, 0.08),
            vib_rms_nde=(1.40, 1.55, 0.08),
            motor_current=(22.5, 22.0, 0.3),
        ),

        # ─────── 阶段 3：泄漏显著，流量穿破 85 ───────
        # 特征：内漏导致泵送能力下降，流量在约 t+45min 处穿破 85
        #       触发 FAL-104（流量低报警）
        Phase(
            state="WARNING",
            duration_sec=7200,   # 2h
            flow_rate=(97.0, 84.0, 0.8),      # 穿过 85
            press_out=(0.305, 0.285, 0.004),  # 出口压力下降
            press_in=(0.018, 0.012, 0.003),   # 入口压力下降（吸入困难）
            temp_de=(46.5, 51.0, 0.5),        # 温度缓慢上升
            temp_nde=(44.0, 47.0, 0.5),
            vib_rms_de=(1.75, 2.30, 0.10),
            vib_rms_nde=(1.55, 1.90, 0.10),
            motor_current=(22.0, 21.0, 0.4),  # 负载下降 → 电流下降
        ),

        # ─────── 阶段 4：润滑油被冲刷，温度穿破 70 ───────
        # 特征：泄漏冲刷轴承润滑油，摩擦增大，温度在约 t+25min 处破 70
        #       触发 TAH-101（温度预警）
        Phase(
            state="WARNING",
            duration_sec=1800,   # 30min
            flow_rate=(84.0, 80.0, 0.8),      # 持续低于 85
            press_out=(0.285, 0.270, 0.004),
            press_in=(0.012, 0.008, 0.003),
            temp_de=(51.0, 72.0, 0.6),        # 快速上升，穿过 70
            temp_nde=(47.0, 58.0, 0.6),
            vib_rms_de=(2.30, 3.20, 0.12),    # 摩擦增大
            vib_rms_nde=(1.90, 2.50, 0.12),
            motor_current=(21.0, 20.0, 0.4),
        ),

        # ─────── 阶段 5：热失控 + 振动超标 ───────
        # 特征：温度穿破 80（TAHH-101），振动穿破 4.5（VAHH-102）
        #       两条一级报警连续触发，随后跳机
        Phase(
            state="WARNING",
            duration_sec=1800,   # 30min
            flow_rate=(80.0, 76.0, 1.0),      # 继续下降
            press_out=(0.270, 0.250, 0.005),
            press_in=(0.008, 0.004, 0.003),
            temp_de=(72.0, 88.0, 0.8),        # 穿过 80 → TAHH-101
            temp_nde=(58.0, 68.0, 0.8),
            vib_rms_de=(3.20, 4.80, 0.15),    # 穿过 4.5 → VAHH-102
            vib_rms_nde=(2.50, 4.00, 0.15),
            motor_current=(20.0, 19.0, 0.5),
        ),

        # ─────── 阶段 6：TRIP 瞬间（20s） ───────
        Phase(
            state="TRIP_SHUTDOWN",
            duration_sec=20,
            flow_rate=(50.0, 0.5, 1.0),
            press_out=(0.155, 0.005, 0.003),
            press_in=(0.010, 0.000, 0.002),
            temp_de=(80.0, 60.0, 1.0),
            temp_nde=(65.0, 55.0, 1.0),
            vib_rms_de=(4.5, 0.5, 0.3),
            vib_rms_nde=(4.3, 0.3, 0.3),
            motor_current=(11.0, 0.1, 0.2),
        ),

        # ─────── 阶段 7：停机冷却（2h） ───────
        Phase(
            state="TRIP_SHUTDOWN",
            duration_sec=7200,
            flow_rate=(0.05, 0.05, 0.03),
            press_out=(0.001, 0.001, 0.002),
            press_in=(0.0, 0.0, 0.002),
            temp_de=(60.0, 25.0, 0.5),
            temp_nde=(55.0, 23.0, 0.5),
            vib_rms_de=(0.1, 0.05, 0.03),
            vib_rms_nde=(0.05, 0.03, 0.02),
            motor_current=(0.05, 0.05, 0.03),
        ),

        # ─────── 阶段 8：启动过渡（30min） ───────
        # 前期流量 < 85 → 触发 FAL-104（状态自动推为 WARNING）
        # 后期流量 > 85 → 恢复 NORMAL
        Phase(
            state="NORMAL",   # 意图 NORMAL，前期会被反推为 WARNING
            duration_sec=1800,
            flow_rate=(0.0, 100.0, 1.0),
            press_out=(0.0, 0.312, 0.005),
            press_in=(0.0, 0.020, 0.002),
            temp_de=(45.0, 45.0, 0.4),
            temp_nde=(43.0, 43.0, 0.4),
            vib_rms_de=(0.1, 1.45, 0.15),
            vib_rms_nde=(0.1, 1.35, 0.15),
            motor_current=(0.0, 22.5, 0.8),
        ),

        # ─────── 阶段 9：修复后稳定运行 ───────
        Phase(state="NORMAL", duration_sec=41380),   # 11.49h，凑齐 24h
    ]

def scenario_bearing_dry() -> List[Phase]:
    """
    轴承干磨完整周期：NORMAL → DEGRADING → WARNING → TRIP → 停机 → 启动 → NORMAL
    """
    return [
        Phase(state="NORMAL", duration_sec=300),

        # 缓升段：温度 45→65，流量仍 100（无报警）
        Phase(
            state="DEGRADING",
            duration_sec=300,
            temp_de=(45.5, 65.0, 0.4),
            temp_nde=(43.0, 52.0, 0.4),
            vib_rms_de=(1.5, 2.5, 0.08),
            vib_rms_nde=(1.4, 2.0, 0.08),
        ),

        # 急剧恶化：温度 65→85，1 分钟（会触发 TAH-101 → TAHH-101）
        Phase(
            state="WARNING",
            duration_sec=60,
            temp_de=(65.0, 85.0, 0.5),
            temp_nde=(52.0, 68.0, 0.5),
            vib_rms_de=(2.5, 4.9, 0.10),
            vib_rms_nde=(2.0, 4.3, 0.10),
        ),

        # 停机瞬间（20s）：各测点骤降
        Phase(
            state="TRIP_SHUTDOWN",
            duration_sec=20,
            flow_rate=(50.0, 0.5, 1.0),
            press_out=(0.155, 0.005, 0.003),
            press_in=(0.010, 0.000, 0.002),
            temp_de=(80.0, 60.0, 1.0),
            temp_nde=(65.0, 55.0, 1.0),
            vib_rms_de=(4.5, 0.5, 0.3),
            vib_rms_nde=(4.3, 0.3, 0.3),
            motor_current=(11.0, 0.1, 0.2),
        ),

        # 停机稳态：流量≈0，温度冷却
        Phase(
            state="TRIP_SHUTDOWN",
            duration_sec=120,
            flow_rate=(0.05, 0.05, 0.03),
            press_out=(0.001, 0.001, 0.002),
            press_in=(0.0, 0.0, 0.002),
            temp_de=(60.0, 45.0, 0.5),
            temp_nde=(55.0, 43.0, 0.5),
            vib_rms_de=(0.1, 0.05, 0.03),
            vib_rms_nde=(0.05, 0.03, 0.02),
            motor_current=(0.05, 0.05, 0.03),
        ),

        # 启动过渡：流量 0 → 100，2 分钟
        # 前期流量<85 → 触发 FAL-104，会被自动推到 WARNING
        # 后期流量>85 → 恢复 NORMAL
        Phase(
            state="NORMAL",  # 意图 NORMAL，但前期会被反推为 WARNING
            duration_sec=120,
            flow_rate=(0.0, 100.0, 1.0),
            press_out=(0.0, 0.312, 0.005),
            press_in=(0.0, 0.020, 0.002),
            temp_de=(45.0, 45.0, 0.4),
            temp_nde=(43.0, 43.0, 0.4),
            vib_rms_de=(0.1, 1.45, 0.15),
            vib_rms_nde=(0.1, 1.35, 0.15),
            motor_current=(0.0, 22.5, 0.8),
        ),

        Phase(state="NORMAL", duration_sec=300),
    ]

def scenario_filter_clog() -> List[Phase]:
    """
    入口滤网堵塞：流量下降，进口压力负值，温度/振动基本平稳。
    """
    return [
        Phase(state="NORMAL", duration_sec=300),

        # 缓堵段：流量 100→85，进口压力降至 -0.01
        Phase(
            state="DEGRADING",
            duration_sec=180,
            flow_rate=(100.0, 85.0, 0.5),
            press_out=(0.312, 0.310, 0.002),
            press_in=(0.020, -0.010, 0.002),
            motor_current=(22.5, 21.0, 0.3),
        ),

        # 报警段：流量 55，进口压力 -0.052
        Phase(
            state="WARNING",
            duration_sec=600,
            flow_rate=(55.0, 55.0, 0.8),
            press_out=(0.328, 0.328, 0.003),
            press_in=(-0.052, -0.052, 0.002),
            temp_de=(45.0, 45.0, 0.4),
            temp_nde=(43.0, 43.0, 0.4),
            vib_rms_de=(1.55, 1.55, 0.08),
            vib_rms_nde=(1.45, 1.45, 0.08),
            motor_current=(16.0, 16.0, 0.3),
        ),

        # 恢复段：流量 55→100，进口压力回升
        # 前期流量<85 → 自动 WARNING；后期流量>85 且压力正常 → NORMAL
        Phase(
            state="NORMAL",  # 意图 NORMAL，前期会被反推为 WARNING
            duration_sec=300,
            flow_rate=(55.0, 100.0, 0.8),
            press_out=(0.328, 0.312, 0.003),
            press_in=(-0.052, 0.020, 0.003),
            motor_current=(16.0, 22.5, 0.4),
        ),

        Phase(state="NORMAL", duration_sec=300),
    ]


def scenario_cavitation() -> List[Phase]:
    """
    气蚀：【Bug 2 修复】加 WARNING 过渡段。
    """
    return [
        Phase(state="NORMAL", duration_sec=300),

        # 气蚀前兆：振动先跳升
        Phase(
            state="WARNING",
            duration_sec=60,
            flow_rate=(100.0, 95.0, 1.5),
            vib_rms_de=(1.5, 4.9, 0.20),
            vib_rms_nde=(1.4, 4.4, 0.20),
        ),

        # 气蚀段：流量剧烈波动
        Phase(
            state="CRITICAL_CAVITATION",
            duration_sec=300,
            flow_rate=(75.0, 75.0, 15.0),
            press_out=(0.25, 0.25, 0.045),
            press_in=(-0.045, -0.045, 0.015),
            vib_rms_de=(5.4, 5.4, 0.30),
            vib_rms_nde=(5.1, 5.1, 0.30),
            motor_current=(19.0, 19.0, 1.5),
        ),

        # 【Bug 2 修复】气蚀修复过渡：30 秒内振动从 5.4 降到 2.5
        Phase(
            state="WARNING",   # 中间过渡段
            duration_sec=60,
            flow_rate=(75.0, 95.0, 1.0),
            press_out=(0.25, 0.31, 0.01),
            press_in=(-0.045, 0.015, 0.005),
            vib_rms_de=(5.4, 2.5, 0.25),
            vib_rms_nde=(5.1, 2.3, 0.25),
            motor_current=(19.0, 22.0, 1.0),
        ),

        Phase(state="NORMAL", duration_sec=300),
    ]


def scenario_unbalance() -> List[Phase]:
    """动不平衡：两端振动同步上升。"""
    return [
        Phase(state="NORMAL", duration_sec=300),

        # 缓升
        Phase(
            state="WARNING",
            duration_sec=300,
            vib_rms_de=(1.5, 5.5, 0.15),
            vib_rms_nde=(1.4, 5.3, 0.15),
            temp_de=(45.0, 53.0, 0.4),
            temp_nde=(43.0, 50.0, 0.4),
        ),

        # 稳态超标
        Phase(
            state="UNBALANCE_MISALIGNMENT",
            duration_sec=300,
            flow_rate=(100.0, 100.0, 0.5),
            vib_rms_de=(5.5, 5.5, 0.20),
            vib_rms_nde=(5.3, 5.3, 0.20),
            temp_de=(53.0, 55.0, 0.4),
            temp_nde=(50.0, 52.0, 0.4),
        ),

        # 修复过渡
        Phase(
            state="WARNING",
            duration_sec=60,
            vib_rms_de=(5.5, 2.0, 0.2),
            vib_rms_nde=(5.3, 1.8, 0.2),
            temp_de=(55.0, 46.0, 0.4),
            temp_nde=(52.0, 44.0, 0.4),
        ),

        Phase(state="NORMAL", duration_sec=300),
    ]

def scenario_realistic_day() -> List[Phase]:
    """
    真实一天：24 小时窗口，仅 1 次轴承干磨故障周期。
    
    时间线（约 24 小时）：
      00:00:00 ~ 07:30:00  NORMAL          7.5h 稳定
      07:30:00 ~ 08:00:00  DEGRADING       30min 温度缓升（未报警）
      08:00:00 ~ 08:05:00  WARNING         5min 破 70℃（TAH-101）
      08:05:00 ~ 08:10:00  WARNING         5min 破 80℃（TAHH-101 + VAHH-102）
      08:10:00 ~ 08:10:20  TRIP_SHUTDOWN   20s 停机
      08:10:20 ~ 10:30:00  TRIP_SHUTDOWN   2h20min 停机冷却
      10:30:00 ~ 11:00:00  启动过渡         30min 前段 WARNING 后段 NORMAL
      11:00:00 ~ 23:59:55  NORMAL          13h 稳定
    
    故障占比：约 40min / 1440min ≈ 2.8%
    """
    return [
        # 7.5h NORMAL
        Phase(state="NORMAL", duration_sec=27000),

        # 30min 缓升（温度 45→68，未破 70 阈值）
        Phase(
            state="DEGRADING",
            duration_sec=1800,
            temp_de=(45.5, 68.0, 0.4),
            temp_nde=(43.0, 52.0, 0.4),
            vib_rms_de=(1.5, 2.8, 0.08),
            vib_rms_nde=(1.4, 2.2, 0.08),
        ),

        # 5min 破 70℃（TAH-101）
        Phase(
            state="WARNING",
            duration_sec=300,
            temp_de=(70.0, 75.5, 0.5),
            temp_nde=(53.0, 58.0, 0.5),
            vib_rms_de=(2.8, 3.6, 0.10),
            vib_rms_nde=(2.2, 3.0, 0.10),
        ),

        # 5min 恶化至 85℃（TAHH-101 + VAHH-102）
        Phase(
            state="WARNING",
            duration_sec=300,
            temp_de=(76.0, 85.0, 0.5),
            temp_nde=(58.0, 68.0, 0.5),
            vib_rms_de=(3.6, 4.9, 0.10),
            vib_rms_nde=(3.0, 4.3, 0.10),
        ),

        # 停机瞬间 20s
        Phase(
            state="TRIP_SHUTDOWN",
            duration_sec=20,
            flow_rate=(50.0, 0.5, 1.0),
            press_out=(0.155, 0.005, 0.003),
            press_in=(0.010, 0.000, 0.002),
            temp_de=(80.0, 60.0, 1.0),
            temp_nde=(65.0, 55.0, 1.0),
            vib_rms_de=(4.5, 0.5, 0.3),
            vib_rms_nde=(4.3, 0.3, 0.3),
            motor_current=(11.0, 0.1, 0.2),
        ),

        # 2h20min 停机冷却
        Phase(
            state="TRIP_SHUTDOWN",
            duration_sec=8380,
            flow_rate=(0.05, 0.05, 0.03),
            press_out=(0.001, 0.001, 0.002),
            press_in=(0.0, 0.0, 0.002),
            temp_de=(60.0, 25.0, 0.5),
            temp_nde=(55.0, 23.0, 0.5),
            vib_rms_de=(0.1, 0.05, 0.03),
            vib_rms_nde=(0.05, 0.03, 0.02),
            motor_current=(0.05, 0.05, 0.03),
        ),

        # 30min 启动过渡（前段流量<85 → WARNING，后段 NORMAL）
        Phase(
            state="NORMAL",
            duration_sec=1800,
            flow_rate=(0.0, 100.0, 1.0),
            press_out=(0.0, 0.312, 0.005),
            press_in=(0.0, 0.020, 0.002),
            temp_de=(45.0, 45.0, 0.4),
            temp_nde=(43.0, 43.0, 0.4),
            vib_rms_de=(0.1, 1.45, 0.15),
            vib_rms_nde=(0.1, 1.35, 0.15),
            motor_current=(0.0, 22.5, 0.8),
        ),

        # 13h NORMAL
        Phase(state="NORMAL", duration_sec=46800),
    ]


def scenario_persistent_alarm() -> List[Phase]:
    """
    故障未修复：持续 WARNING 状态数小时（模拟等排班维修）。
    
    时间线（4 小时）：
      00:00:00 ~ 01:00:00  NORMAL      1h
      01:00:00 ~ 01:30:00  DEGRADING   30min
      01:30:00 ~ 03:30:00  WARNING     2h 持续报警，未修复
      03:30:00 ~ 04:30:00  NORMAL      1h 恢复
    """
    return [
        Phase(state="NORMAL", duration_sec=3600),
        Phase(
            state="DEGRADING",
            duration_sec=1800,
            temp_de=(45.5, 68.0, 0.4),
            temp_nde=(43.0, 52.0, 0.4),
        ),
        # 2h 稳定在报警区间（不是继续恶化，也不是恢复）
        Phase(
            state="WARNING",
            duration_sec=7200,
            temp_de=(72.0, 78.0, 0.4),      # 稳定在 72~78℃
            temp_nde=(56.0, 62.0, 0.4),
            vib_rms_de=(3.8, 4.2, 0.10),
            vib_rms_nde=(3.0, 3.5, 0.10),
        ),
        Phase(state="NORMAL", duration_sec=3600),
    ]

def scenario_stress_test() -> List[Phase]:
    """
    ⚠ 压力测试场景：83 分钟塞入 4 类故障，故障密度远超真实工业场景。
    
    真实场景参考：单台泵 MTBF 约 8000~20000h，故障占比 < 2%。
    此场景的故障占比约 70%，**仅用于回归测试**分段/报警码提取能力，
    不应作为 LLM 描述长度或统计特征的真实基准。
    
    时间线（约 83 分钟）：
      08:00:00 ~ 08:04:55  NORMAL
      08:05:00 ~ 08:10:10  DEGRADING（温度 45→69）
      08:10:15 ~ 08:10:55  WARNING（TAH→TAHH→VAHH）
      08:11:00 ~ 08:13:15  TRIP_SHUTDOWN
      08:13:20 ~ 08:15:00  WARNING（启动过渡）
      08:15:05 ~ 08:25:15  NORMAL
      08:25:20 ~ 08:28:15  DEGRADING（滤网堵塞前期）
      08:28:20 ~ 08:41:45  WARNING（滤网堵塞）
      08:41:50 ~ 08:53:50  NORMAL
      08:53:55 ~ 08:54:15  WARNING（气蚀前兆）
      08:54:20 ~ 08:59:15  CRITICAL_CAVITATION
      08:59:20 ~ 08:59:55  WARNING
      09:00:00 ~ 09:12:30  NORMAL
      09:12:35 ~ 09:15:15  WARNING（动不平衡前兆 + 单点碎片）
      09:15:20 ~ 09:20:15  UNBALANCE_MISALIGNMENT
      09:20:20 ~ 09:20:55  WARNING
      09:21:00 ~ 09:23:15  NORMAL
    """
    phases: List[Phase] = []
    phases.extend(scenario_bearing_dry())
    phases.extend(scenario_filter_clog())
    phases.extend(scenario_cavitation())
    phases.extend(scenario_unbalance())
    return phases


SCENARIOS = {
    # 单场景
    "normal":           scenario_normal,
    "bearing_dry":      scenario_bearing_dry,
    "filter_clog":      scenario_filter_clog,
    "cavitation":       scenario_cavitation,
    "unbalance":        scenario_unbalance,
    "seal_leak":        scenario_seal_leak, 
    # 长时间线（真实密度）
    "realistic_day":    scenario_realistic_day,
    "persistent_alarm": scenario_persistent_alarm,
    # 压力测试（高故障密度，非现实）
    "stress_test":      scenario_stress_test,
}


# ==================== 持久化 ====================

def persist(rows: List[dict], device_id: str, clean: bool = False) -> int:
    db = session_creater()
    try:
        if clean:
            print(f"[DB] 清空设备 {device_id} 的历史数据 ...")
            db.execute(delete(ScadaTelemetry).where(ScadaTelemetry.device_id == device_id))
            db.commit()

        objs = [
            ScadaTelemetry(
                timestamp=r["timestamp"],
                device_id=device_id,
                flow_rate=r["flow_rate"],
                press_out=r["press_out"],
                press_in=r["press_in"],
                temp_de=r["temp_de"],
                temp_nde=r["temp_nde"],
                vib_rms_de=r["vib_rms_de"],
                vib_rms_nde=r["vib_rms_nde"],
                motor_current=r["motor_current"],
                operating_state=OperatingState(r["operating_state"]),
                alarm_code=r["alarm_code"],
            )
            for r in rows
        ]
        db.add_all(objs)
        db.commit()
        return len(objs)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ==================== CLI ====================

def _parse_start(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", ""))


def _summarize(rows: List[dict]) -> None:
    """打印统计 + Bug 检查。"""
    if not rows:
        print("[!] 未生成任何数据")
        return

    print(f"\n{'=' * 70}")
    print(f"生成数据统计：共 {len(rows)} 条，时间跨度 "
          f"{rows[0]['timestamp']} ~ {rows[-1]['timestamp']}")
    print(f"{'=' * 70}")

    # 按状态统计
    from itertools import groupby
    state_summary = {}
    for r in rows:
        s = r["operating_state"]
        state_summary.setdefault(s, {"count": 0, "first": r["timestamp"], "last": r["timestamp"]})
        state_summary[s]["count"] += 1
        state_summary[s]["last"] = r["timestamp"]

    print("\n状态分布：")
    for state, info in state_summary.items():
        print(f"  [{state:24s}]  {info['count']:4d} 条  "
              f"{info['first']} ~ {info['last']}")

    # 报警码统计
    alarm_counter: dict[str, int] = {}
    for r in rows:
        for code in r["alarm_code"].split(";"):
            code = code.strip()
            if code and code != "NONE":
                alarm_counter[code] = alarm_counter.get(code, 0) + 1
    print()
    if alarm_counter:
        print("报警码命中次数：")
        for code, cnt in sorted(alarm_counter.items()):
            print(f"    {code:12s}: {cnt} 次")
    else:
        print("报警码：无")

    # 【新增】一致性检查：状态 vs 报警码
    print(f"\n{'=' * 70}")
    print("一致性检查（状态与报警码是否矛盾）：")
    print(f"{'=' * 70}")

    inconsistent = []
    for r in rows:
        state = r["operating_state"]
        alarm = r["alarm_code"]

        # NORMAL 段不应有非 NONE 报警码
        if state == "NORMAL" and alarm != "NONE":
            inconsistent.append((r["timestamp"], state, alarm, "NORMAL 段不应有报警码"))

        # TRIP_SHUTDOWN 段报警码应为 NONE
        if state == "TRIP_SHUTDOWN" and alarm != "NONE":
            inconsistent.append((r["timestamp"], state, alarm, "停机段报警码应为 NONE"))

    if inconsistent:
        print(f"⚠ 发现 {len(inconsistent)} 条不一致：")
        for ts, state, alarm, reason in inconsistent[:10]:
            print(f"  {ts}  state={state}  alarm={alarm}  → {reason}")
    else:
        print("✅ 无矛盾（状态与报警码一致）")

    print(f"{'=' * 70}\n")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="生成工业离心泵 SCADA 时序数据并写入 MySQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "可用场景：",
            "  normal            纯正常工况",
            "  bearing_dry       轴承干磨（默认）",
            "  filter_clog       入口滤网堵塞",
            "  cavitation        气蚀",
            "  unbalance         转子动不平衡",
            "  seal_leak         机械密封泄漏（24h 完整周期）",   
            "  realistic_day     真实一天（24h，1 次故障）",
            "  persistent_alarm  故障未修复（持续报警数小时）",
            "  stress_test       ⚠ 压力测试（83min 4 类故障，非现实密度）",
        ]),
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE_ID,
                        help=f"设备 ID（默认：{DEFAULT_DEVICE_ID}）")
    parser.add_argument("--start", default=DEFAULT_START_TIME,
                        help=f"起始时间 ISO 格式（默认：{DEFAULT_START_TIME}）")
    parser.add_argument("--scenario", default="bearing_dry",
                        choices=list(SCENARIOS.keys()),
                        help="场景名（默认：bearing_dry）")
    parser.add_argument("--duration", type=int, default=30,
                        help="normal 场景的持续分钟数（默认：30）")
    parser.add_argument("--clean", action="store_true",
                        help="插入前清空该设备的历史数据")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（用于复现）")

    args = parser.parse_args(argv)

    if args.seed is not None:
        np.random.seed(args.seed)

    start_dt = _parse_start(args.start)
    scenario_fn = SCENARIOS[args.scenario]

    if args.scenario == "normal":
        phases = scenario_fn(duration_min=args.duration)
    else:
        phases = scenario_fn()

    print(f"[Gen] 设备={args.device}  场景={args.scenario}  起点={start_dt}")
    rows = generate_scenario(phases, start_dt)
    _summarize(rows)

    n = persist(rows, device_id=args.device, clean=args.clean)
    print(f"[DB] 已写入 {n} 条记录到 scada_telemetry\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())