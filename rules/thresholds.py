"""确定性规则资产：全部数值判据的唯一出处。

为什么必须集中管理
==================
1. **同一物理量在不同标准中限值不同**（轴承温度：GB 50275 上限 80℃，
   IOM 文本写 85℃），现场必须以「更严者优先」并显式标注出处，
   不能散落在判断语句里靠人记。
2. **Step 2 的规则判定与 Step 6 的安全门禁共用同一批阈值**。
   放两处必然漂移 —— 一处调了另一处没调，两份结论就打架。
3. 所有判定结论都要回填出处（rule_id / source），供报告 100% 溯源。

取值依据
========
《IS100-80-160 工业离心泵 SCADA 时序数据标准与异常诊断规范》
  · 第三节：标准数据运行范围与安全阈值（绿区 / 预警线 / 停机线）
  · 第五节：指标计算与判定标准（偏离度 / 恶化斜率 / 离散度）
  · 第六节：ISA-5.1 工业报警代码触发映射表
以及 GB 50275-2010、GB/T 3216-2016、IOM 规程。

单位约定
========
温度恶化斜率内部一律用 ℃/min（与规范 5.2 节一致）。
"""

from __future__ import annotations

from typing import Final

# =============================================================================
# 一、额定工况基准值（规范 1.2 节静态台账 / 第三节额定基准值）
# =============================================================================

RATED_FLOW_M3H: Final[float] = 100.0          # 瞬时输送流量 m³/h
RATED_PRESS_OUT_MPA: Final[float] = 0.312     # 泵出口压力 MPa
RATED_PRESS_IN_MPA: Final[float] = 0.020      # 泵入口压力 MPa
RATED_TEMP_C: Final[float] = 45.0             # 轴承温度 ℃
RATED_VIB_MM_S: Final[float] = 1.45           # 振动速度有效值 mm/s
RATED_CURRENT_A: Final[float] = 22.5          # 电机三相运行电流 A

# =============================================================================
# 二、规范第三节：安全阈值（绿区 / 预警线 / 停机线）
# =============================================================================

# —— 轴承温度（GB 50275-2010：滚动轴承 ≤80℃）——
TEMP_GREEN_LOW_C: Final[float] = 35.0
TEMP_GREEN_HIGH_C: Final[float] = 65.0
TEMP_WARN_C: Final[float] = 70.0              # 预警线 ≥70℃
TEMP_TRIP_C: Final[float] = 80.0              # 报警停机线 ≥80℃

# —— 振动（GB 50275 附录 A：振动 ≤4.5 mm/s）——
VIB_GREEN_LOW_MM_S: Final[float] = 0.50
VIB_GREEN_HIGH_MM_S: Final[float] = 2.80
VIB_WARN_MM_S: Final[float] = 3.50            # 良好区上限
VIB_TRIP_MM_S: Final[float] = 4.50            # 超国标限值

# —— 流量（GB/T 3216 容差 ±9%）——
FLOW_GREEN_LOW_M3H: Final[float] = 91.0
FLOW_GREEN_HIGH_M3H: Final[float] = 109.0
FLOW_WARN_LOW_M3H: Final[float] = 85.0

# —— 出口压力（扬程容差 ±7%）——
PRESS_OUT_GREEN_LOW_MPA: Final[float] = 0.290
PRESS_OUT_GREEN_HIGH_MPA: Final[float] = 0.334

# —— 入口压力（允许吸上高度 5.5m，防汽化汽蚀）——
PRESS_IN_GREEN_LOW_MPA: Final[float] = -0.020
PRESS_IN_GREEN_HIGH_MPA: Final[float] = 0.050
PRESS_IN_WARN_MPA: Final[float] = -0.040      # 预警线
PRESS_IN_TRIP_MPA: Final[float] = -0.055      # 停机线

# —— 电机电流（Y160M2-2 额定 29A）——
CURRENT_GREEN_LOW_A: Final[float] = 18.0
CURRENT_GREEN_HIGH_A: Final[float] = 26.0
CURRENT_LOW_A: Final[float] = 16.0            # 欠载（规范第六节 IAL-105）
CURRENT_HIGH_A: Final[float] = 29.0           # 额定满载跳闸（IAH-106）

# =============================================================================
# 三、规范第五节：计算类判据
# =============================================================================

# —— 5.1 工况偏离度 ——
DEV_FLOW_PCT: Final[float] = 15.0             # |Dev_flow| > 15% → 流量工况漂移
DEV_PRESS_PCT: Final[float] = 20.0            # |Dev_press| > 20% → 水力压头异常

# —— 5.2 恶化斜率（℃/min）——
TEMP_SLOPE_SLOW: Final[float] = 0.2           # 0.2 ≤ k < 0.8：缓慢劣化（黄色关注）
TEMP_SLOPE_SHARP: Final[float] = 0.8          # k ≥ 0.8：急剧恶化（红色紧急，高危干磨）

# —— 5.3 离散度（CV 变异系数）——
CV_CAVITATION_PCT: Final[float] = 12.0        # CV ≥ 12.0% 判流态失稳/气蚀脱流
CV_WINDOW_SEC: Final[int] = 120               # 滚动窗长度：连续 2 分钟
CV_MIN_POINTS: Final[int] = 12                # 窗内最少点数（5s 采样 × 12 = 60s）

# =============================================================================
# 四、判定用的上下文门槛（工程加固，实测踩坑后加的）
# =============================================================================

# 温度绝对值门槛：低于此值不判斜率。
# 依据：常温段（45→55℃）的斜率没有诊断意义，只有接近预警线的爬升才值得报。
TEMP_SLOPE_ACTIVATION_C: Final[float] = 60.0
# 斜率判定的最短窗口与最少点数，防止极短窗口被噪声顶穿阈值
TEMP_SLOPE_MIN_DURATION_SEC: Final[int] = 180
TEMP_SLOPE_MIN_POINTS: Final[int] = 36
# 1 分钟滑窗对应的点数（5s 采样）
SLOPE_WINDOW_POINTS: Final[int] = 12

# 流量低于此值视为「停机段」，不参与流量偏离 / CV 告警。
# 依据：停机时流量归零，若照常判定会凭空产生"流量低/欠载"的假象。
FLOW_ACTIVE_THRESHOLD_M3H: Final[float] = 20.0

# 两端振动「同步偏高」的判据与对照值（规范第四节第 4 类：转子动不平衡/不对中）
VIB_SYNC_HIGH_MM_S: Final[float] = 5.0        # 两端都 > 5.0 才谈得上"同步"
VIB_SYNC_LOW_MM_S: Final[float] = 3.0         # 对照：另一端 < 3.0 判单端显著偏高

# 参与告警判定的「活跃状态」。
#
# ★ 注意：这是 **兜底白名单**，不是"用 SCADA 状态标签豁免规则"。
#   它的用途是排除 TRIP_SHUTDOWN —— 停机是故障的**结果**，
#   其低流量/低振动/低电流不是异常源，照常判定会报一堆假告警。
#   规则本身只依赖物理量，不依赖 operating_state 的正确性。
ACTIVE_STATES: Final[frozenset[str]] = frozenset({
    "NORMAL", "DEGRADING", "WARNING",
    "CRITICAL_CAVITATION", "UNBALANCE_MISALIGNMENT",
})

# =============================================================================
# 四之二、数据完整性判据（用于 DataQuality.status = "PARTIAL"）
# =============================================================================

# SCADA 采样周期（秒）。现有数据集是 5s 一个点。
SAMPLE_INTERVAL_SEC: Final[int] = 5

# 数据覆盖率下限（%）。窗口内实际点数低于"理论点数 × 此比例"即判数据不完整。
# 依据：缺个把点是正常的，缺成片就该让下游（Step 6 安全门禁）知道这份分析不可全信。
DATA_COVERAGE_MIN_PCT: Final[float] = 90.0

# 阈值档案标识：写入报告供溯源（改阈值务必同步改版本号）
THRESHOLD_PROFILE_ID: Final[str] = "SCADA-SPEC-2026-V1 + GB50275-2010"


# =============================================================================
# 五、规则码表（rule_id → 人读标签 + 出处）
#
# 用途：
#   1. Step 2 的 data.threshold_flags 只存**机器码**（短、稳定、可溯源）；
#   2. 拼大模型提示词时，由 render_rule_hits 把码翻回中文事实句（只活在 prompt 里）；
#   3. 下游（Step 6 门禁 / 报告溯源）直接吃码，不用正则解析中文。
#
# 命名规范（本项目自定）：<测点>_<位置>_<性质>
#   · 测点：BEARING_TEMP / VIB / FLOW / PRESS / CURRENT / CV
#   · 位置：DE（驱动端）/ NDE（非驱动端）/ 省略
#   · 性质：TRIP / WARN / SHARP / SLOW / HIGH / LOW / DOMINANT / DEV / UNSTABLE
#
# ★ 这不是 SCADA 的 ISA-5.1 报警码（TAH-101 那一套）。
#   那套是数据里的**报警事实**，归 data.alarms；本表是 Step 2 **自算判据**的码。
#
# ★ extra_field / extra_template：少数规则额外要附一句上下文（如"约从 08:01:00 开始"），
#   渲染器会从该 phase 里取 extra_field 的值填进模板，取不到就不加。
# =============================================================================

RULE_CATALOG: Final[dict[str, dict]] = {
    # —— 温度（只判驱动端）——
    "BEARING_TEMP_DE_TRIP": dict(
        label="驱动端温度超停机线", source="GB 50275-2010 / 规范第三节"),
    "BEARING_TEMP_DE_WARN": dict(
        label="驱动端温度超预警线", source="规范第三节"),
    "BEARING_TEMP_DE_RAMP_SHARP": dict(
        label="驱动端温度急剧恶化", source="规范 5.2",
        extra_field="ramp_start_time", extra_template="，约从 {v} 开始"),
    "BEARING_TEMP_DE_RAMP_SLOW": dict(
        label="驱动端温度缓慢劣化", source="规范 5.2",
        extra_field="ramp_start_time", extra_template="，约从 {v} 开始"),
    # —— 振动 ——
    "VIB_DE_TRIP": dict(label="驱动端振动超国标停机线", source="GB 50275 附录 A"),
    "VIB_DE_WARN": dict(label="驱动端振动超良好区上限", source="GB 50275 附录 A"),
    "VIB_NDE_TRIP": dict(label="非驱动端振动超国标停机线", source="GB 50275 附录 A"),
    "VIB_NDE_WARN": dict(label="非驱动端振动超良好区上限", source="GB 50275 附录 A"),
    "VIB_BOTH_HIGH": dict(label="两端振动同步偏高", source="规范第四节"),
    "VIB_DE_DOMINANT": dict(label="驱动端振动显著高于非驱动端", source="规范第四节"),
    "VIB_NDE_DOMINANT": dict(label="非驱动端振动显著高于驱动端", source="规范第四节"),
    # —— 水力 ——
    "FLOW_DEV": dict(label="流量工况偏离额定值", source="规范 5.1"),
    "PRESS_OUT_DEV": dict(label="出口压力工况偏离额定值", source="规范 5.1"),
    "PRESS_IN_LOW": dict(label="入口压力低", source="规范第三节 / 第六节 PAL-103"),
    "CV_UNSTABLE": dict(label="流量/压力波动超离散度阈值", source="规范 5.3"),
    # —— 电气 ——
    "CURRENT_HIGH": dict(label="电机电流超额定满载", source="规范第六节 IAH-106"),
    "CURRENT_LOW": dict(label="电机电流欠载", source="规范第六节 IAL-105"),
}
