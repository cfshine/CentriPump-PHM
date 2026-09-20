# tests/test_pipeline.py
"""Step 2 端到端回归测试（30 个用例）。

本文件是从原 ``analysisAgent/graph.py`` 里抽出来的测试台 —— 它原本和业务代码
混在一个文件里（638 行里 610 行是测试），导致要复用业务逻辑得先读 600 行测试。

用法（需要 MySQL + 大模型 Key，Key 由 langchain 从环境变量 DEEPSEEK_API_KEY 读取）::

    python -m tests.test_pipeline

★ 断言口径：绑**物理量**（斜率、点数、段数、报警码集合）而不是绑死某次生成的数据。
  历史教训：同一套用例在同一份代码上跑出过 26/27、9/30、27/30 三种结果，
  根因就是断言绑了"数据指纹"，换一批生成数据就整体崩。

★ 2026-09-20 适配组长 9 盒子契约后的口径变化（本文件已同步）：
  · 子图仍是 2 节点（analyze + summarize），调用方式不变；
  · 产出改落在 ``final["data"]``（DataState）里：
    指标取 ``data.metrics``、规则码取 ``data.threshold_flags``、
    报警码取 ``data.alarms.effective``、质量取 ``data.quality.status``、
    描述取 ``data.descriptions``（当前是 1 条 type=OTHER，第 3 步改成多条分类）；
  · 初始状态必须用组长的 ``create_initial_state(...)`` 构造（start/end 是必填的）；
  · **降级改用 ``quality_status`` / ``quality_reason_contains`` 断言**，
    不再检查"无数据"这类中文告警（告警位现在只放规则码）；
  · **E2 的行为**：入参 alarm_code 不再被采用 —— 窗口内没有报警就是 NONE。
"""
from src.orchestrator.graph import data_agent_graph as step2_agent
from src.schemas.state import create_initial_state

# ============================================================
# 测试用例 - 适配迟滞版 SCADA 生成脚本
# ============================================================
# 数据时间线（约 83 分钟）：
#   08:00:00 ~ 08:04:55  NORMAL
#   08:05:00 ~ 08:09:55  DEGRADING
#   08:10:00 ~ 08:10:10  NORMAL (10s 过渡)
#   08:10:15 ~ 08:10:55  WARNING (温度急剧恶化)
#   08:11:00 ~ 08:13:15  TRIP_SHUTDOWN
#   08:13:20 ~ 08:14:55  WARNING (启动过渡)
#   08:15:00 ~ 08:25:15  NORMAL
#   08:25:20 ~ 08:28:15  DEGRADING
#   08:28:20 ~ 08:41:35  WARNING (滤网堵塞)
#   08:41:40 ~ 08:41:45  WARNING + NONE (迟滞尾)
#   08:41:50 ~ 08:53:50  NORMAL
#   08:53:55 ~ 08:54:15  WARNING (气蚀前兆)
#   08:54:20 ~ 08:59:15  CRITICAL_CAVITATION
#   08:59:20 ~ 08:59:55  WARNING (气蚀过渡)
#   09:00:00 ~ 09:12:35  NORMAL
#   09:12:35 ~ 09:15:15  WARNING (动不平衡前兆，中间夹 1 个 NORMAL 碎片)
#   09:15:20 ~ 09:20:20  UNBALANCE_MISALIGNMENT
#   09:20:20 ~ 09:20:55  WARNING (修复过渡)
#   09:21:00 ~ 09:23:15  NORMAL
# ============================================================


def _build_state(device_id, start, end, alarm_code=""):
    """构造初始 State（用组长契约里的 ``create_initial_state``）。

    ★ 2026-09-20 起公共契约是 9 个嵌套盒子，**必须**走组长的构造函数：
      ``start_time`` / ``end_time`` 在他的签名里是必填的（模拟"没给窗口"要显式传空串）。

    ``alarm_code`` 仍然保留在签名里（Step 3 还用它），但 Step 2 **不再读它**：
    报警码一律以窗口内数据为准，E2 用例专门守护这件事。
    """
    return create_initial_state(
        trace_id=f"test-{device_id}-{start}",
        device_id=device_id,
        start_time=start,
        end_time=end,
        alarm_code=alarm_code,
    )


# ============================================================
# 预期校验（容错版）
# ============================================================
def _check_expect(expect, actual):
    """
    统一的预期校验。
    expect 里所有字段均为可选；值为 None 视为"跳过校验"。
    返回 (通过项列表, 失败项列表)。
    """
    if not expect:
        return [], []

    passed, failed = [], []

    # ---- 全局结构不变量（每条用例都查）：descriptions 的条数与 evidence 可回溯性 ----
    if actual["quality_status"] == "EMPTY":
        # 降级窗口（没窗口/没数据）：按设计**不生成描述、也不调大模型**
        if actual["desc_count"] == 0:
            passed.append("降级窗口：0 条描述（按设计不生成）")
        else:
            failed.append(f"降级窗口不该有描述，实际 {actual['desc_count']} 条")
    elif not (1 <= actual["desc_count"] <= 6):
        failed.append(f"描述条数 {actual['desc_count']} 超出契约的 1~6")
    elif actual["bad_evidence"]:
        failed.append(f"有 evidence 无法回溯到材料：{actual['bad_evidence']}")
    else:
        passed.append(f"描述 {actual['desc_count']} 条 {'/'.join(actual['desc_types'])}，evidence 均可回溯")

    # ---- 精确断言 ----
    for key in ("points", "phases", "alarm_count", "states"):
        exp = expect.get(key)
        if exp is not None:
            actual_key = {"points": "points",
                          "phases": "phase_count",
                          "alarm_count": "alarm_count",
                          "states": "states"}[key]
            if exp == actual[actual_key]:
                passed.append(f"{key}={exp}")
            else:
                failed.append(f"期望 {key}={exp}，实际={actual[actual_key]}")

    # ---- 范围断言 ----
    for key, actual_key in (("points_min", "points"), ("points_max", "points"),
                             ("phases_min", "phase_count"), ("phases_max", "phase_count")):
        exp = expect.get(key)
        if exp is not None:
            v = actual[actual_key]
            if key.endswith("_min"):
                if v >= exp:
                    passed.append(f"{key}≥{exp}（实际 {v}）")
                else:
                    failed.append(f"期望 {key}≥{exp}，实际={v}")
            else:
                if v <= exp:
                    passed.append(f"{key}≤{exp}（实际 {v}）")
                else:
                    failed.append(f"期望 {key}≤{exp}，实际={v}")

    # ---- alarm_count_min ----
    exp = expect.get("alarm_count_min")
    if exp is not None:
        if actual["alarm_count"] >= exp:
            passed.append(f"alarm_count≥{exp}（实际 {actual['alarm_count']}）")
        else:
            failed.append(f"期望 alarm_count≥{exp}，实际={actual['alarm_count']}")

    # ---- has_alarm ----
    exp = expect.get("has_alarm")
    if exp is not None:
        actual_has = actual["alarm_count"] > 0
        if exp == actual_has:
            passed.append(f"has_alarm={exp}")
        else:
            failed.append(f"期望 has_alarm={exp}，实际={actual_has}")

    # ---- states_contains（顺序子序列） ----
    exp = expect.get("states_contains")
    if exp is not None:
        it = iter(actual["states"])
        missing = [s for s in exp if s not in it]
        if not missing:
            passed.append(f"states 按序包含 {exp}")
        else:
            failed.append(f"states 缺少: {missing}（实际 {actual['states']}）")

    # ---- states_set_contains（无序集合） ----
    exp = expect.get("states_set_contains")
    if exp is not None:
        actual_set = set(actual["states"])
        missing = [s for s in exp if s not in actual_set]
        if not missing:
            passed.append(f"states 集合包含 {exp}")
        else:
            failed.append(f"states 集合缺少: {missing}（实际 {sorted(actual_set)}）")

    # ---- effective_alarm_codes ----
    # ★ 取值口径：窗口内出现过的报警码，来自 calculated_metrics.overall
    #   （2026-09-17 精简：顶层 effective_alarm_codes / last_alarm_codes /
    #     all_alarm_codes_in_window 三个字段已删除，只保留 overall 里这一处）
    exp = expect.get("effective_alarm_codes")
    if exp is not None:
        if exp == actual["effective_alarm_codes"]:
            passed.append(f"effective_alarm_codes={exp}")
        else:
            failed.append(
                f"期望 effective_alarm_codes={exp}，实际={actual['effective_alarm_codes']}"
            )

    # ---- effective_alarm_codes_contains ----
    exp = expect.get("effective_alarm_codes_contains")
    if exp is not None:
        actual_codes = set(
            c.strip()
            for c in actual["effective_alarm_codes"].split(";")
            if c.strip() and c.strip() != "NONE"
        )
        missing = [c for c in exp if c not in actual_codes]
        if not missing:
            passed.append(f"effective_alarm_codes 包含 {exp}")
        else:
            failed.append(f"effective_alarm_codes 缺少: {missing}")

    # ---- effective_alarm_codes_excludes ----
    exp = expect.get("effective_alarm_codes_excludes")
    if exp is not None:
        actual_codes = set(
            c.strip()
            for c in actual["effective_alarm_codes"].split(";")
            if c.strip()
        )
        unexpected = [c for c in exp if c in actual_codes]
        if not unexpected:
            passed.append(f"effective_alarm_codes 不含 {exp}")
        else:
            failed.append(f"effective_alarm_codes 意外包含: {unexpected}")

    # ---- alarm_contains ----
    # ★ 2026-09-20 起 threshold_flags 存的是**机器码**（如 BEARING_TEMP_DE_TRIP），
    #   所以这里比对的是码，不再是中文句子。
    exp = expect.get("alarm_contains")
    if exp is not None:
        all_flags = " | ".join(actual["alarm_flags"])
        missing = [s for s in exp if s not in all_flags]
        if not missing:
            passed.append(f"alarm_contains={exp}")
        else:
            failed.append(f"告警中缺少子串: {missing}")

    # ---- alarm_excludes ----
    exp = expect.get("alarm_excludes")
    if exp is not None:
        all_flags = " | ".join(actual["alarm_flags"])
        unexpected = [s for s in exp if s in all_flags]
        if not unexpected:
            passed.append(f"alarm_excludes={exp}")
        else:
            failed.append(f"告警中意外包含: {unexpected}")

    # ---- no_rule_hits_on_states ----
    # ★ 替代原来的 `alarm_excludes: ["[NORMAL段]"]`：码化以后，"哪一段命中了什么"
    #   已经落在 phases[*].rule_hits 上，所以可以**直接断言某些状态上不许有命中**，
    #   比原来"看告警文案里有没有 [NORMAL段] 前缀"更准。
    exp = expect.get("no_rule_hits_on_states")
    if exp is not None:
        offenders = [(s, h) for s, h in actual["phases_rule_hits"] if s in exp and h]
        if not offenders:
            passed.append(f"这些状态上无规则命中: {exp}")
        else:
            failed.append(f"这些状态上不该有规则命中，实际: {offenders}")

    # ---- quality_status / quality_reason_contains ----
    # ★ 2026-09-20：降级不再靠"中文告警文案"，改由 data.quality（组长的 DataQuality）表达
    exp = expect.get("quality_status")
    if exp is not None:
        if exp == actual["quality_status"]:
            passed.append(f"quality_status={exp}")
        else:
            failed.append(f"期望 quality_status={exp}，实际={actual['quality_status']}")

    exp = expect.get("quality_reason_contains")
    if exp is not None:
        missing = [s for s in exp if s not in actual["quality_reason"]]
        if not missing:
            passed.append(f"quality_reason 含 {exp}")
        else:
            failed.append(f"quality_reason 缺少: {missing}（实际：{actual['quality_reason']}）")

    # ---- expect_data_bug ----
    if expect.get("expect_data_bug"):
        if actual["alarm_count"] > 0:
            passed.append("⚠ 数据 bug 已被 Step 2 检出（符合预期）")
        else:
            failed.append("预期应检出数据 bug，但告警数为 0（可能 bug 已修）")
    
    # ---- max_desc_len / min_desc_len ----
    exp_max = expect.get("max_desc_len")
    if exp_max is not None:
        actual_len = actual.get("llm_desc_len", 0)
        if actual_len <= exp_max:
            passed.append(f"llm_desc_len≤{exp_max}（实际 {actual_len}）")
        else:
            failed.append(f"期望 llm_desc_len≤{exp_max}，实际={actual_len}")

    exp_min = expect.get("min_desc_len")
    if exp_min is not None:
        actual_len = actual.get("llm_desc_len", 0)
        if actual_len >= exp_min:
            passed.append(f"llm_desc_len≥{exp_min}（实际 {actual_len}）")
        else:
            failed.append(f"期望 llm_desc_len≥{exp_min}，实际={actual_len}")

    return passed, failed


# ============================================================
# 用例执行器
# ============================================================
def _run_case(case_id, name, device_id, start, end, alarm_code="", expect=None):
    print(f"\n{'━' * 80}")
    print(f"[{case_id}] {name}")
    print(f"     窗口: {start} ~ {end}")
    print(f"     报警码: {alarm_code or 'NONE'}")
    if expect:
        print(f"     说明: {expect.get('note', '')}")
    print(f"{'━' * 80}")

    try:
        final = step2_agent.invoke(_build_state(device_id, start, end, alarm_code).model_dump())
    except Exception as e:
        print(f"❌ 执行异常: {type(e).__name__}: {e}")
        return None, False

    # ★ 2026-09-20：产出落在 data 盒子里（不再是顶层扁平字段）
    data = final["data"]
    metrics = data.metrics
    overall = metrics.get("overall", {})
    phases = metrics.get("phases", [])
    flags = data.threshold_flags

    actual = {
        "points": overall.get("total_points", 0),
        "phase_count": overall.get("phase_count", 0),
        # ★ alarm_count 现在是**去重后的规则码种数**（不再是"告警行数"）
        "alarm_count": len(flags),
        "alarm_flags": flags,
        "states": [ph["state"] for ph in phases],
        # 段级规则命中：[(状态, [码...]), ...]，供 no_rule_hits_on_states 断言
        "phases_rule_hits": [(ph["state"], ph.get("rule_hits", [])) for ph in phases],
        # 报警码的唯一出处：data.alarms（组长契约的三件套之一）
        "effective_alarm_codes": data.alarms.effective,
        "quality_status": data.quality.status,
        "quality_reason": data.quality.reason,
        "start_state": overall.get("start_state", "?"),
        "end_state": overall.get("end_state", "?"),
        "llm_desc_len": sum(len(d.description) for d in data.descriptions),
        # descriptions 的结构信息（type 六选一、evidence 必须能回溯到材料）
        "desc_count": len(data.descriptions),
        "desc_types": [d.type for d in data.descriptions],
        "bad_evidence": [
            e for d in data.descriptions for e in d.evidence
            if e not in (set(overall) | {k for ph in phases for k in ph} | set(flags))
        ],
    }

    # ============ 打印 ============
    print(f"数据点数:   {actual['points']}")
    print(f"阶段数:     {actual['phase_count']}")
    print(f"起始→结束:  {actual['start_state']} → {actual['end_state']}")
    print(f"数据质量:   {actual['quality_status']}  {actual['quality_reason']}")
    print(f"描述:       {actual['desc_count']} 条  {'/'.join(actual['desc_types'])}")
    print(f"窗口内报警码: {actual['effective_alarm_codes']}")
    print(f"告警数:     {actual['alarm_count']}")

    for ph in phases:
        t0 = ph.get("start", "")[-19:] if ph.get("start") else ""
        t1 = ph.get("end", "")[-19:] if ph.get("end") else ""
        print(f"  ◆ [{ph['state']:24s}] {ph['duration_sec']:4d}s  {t0} ~ {t1}")
        print(f"        temp_slope={ph['slope_temp_de']:>7}  "
              f"max_temp={ph['max_temp_de']:>6}  "
              f"vib_max={ph['max_vib_de']:>6}  "
              f"cv_flow={ph['cv_flow']}")

    for f in flags:
        print(f"    ⚠ {f}")

    print(f"LLM 描述长度: {actual['llm_desc_len']}")

    # ============ 预期核对 ============
    all_passed = True
    if expect:
        passed, failed = _check_expect(expect, actual)
        for p in passed:
            print(f"  ✅ {p}")
        for f in failed:
            print(f"  ❌ {f}")
            all_passed = False

    return final, all_passed


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # ================================================================
    # 数据集：PUMP-IS100-80-160-01
    #   2026-09-13 00:00:00 ~ 2026-09-14 23:59:55
    #   5s 采样，两天连续，共 34560 点
    #   场景线：密封泄漏 → 流量下降(FAL-104 抖动) → 轴承温度/振动升高
    #           → 联锁停机 → 重启欠载(IAL-105) → 恢复
    #
    #   ★ 重要覆盖说明：本数据集**不含**以下两类状态与一个报警码 ——
    #       CRITICAL_CAVITATION（气蚀）、UNBALANCE_MISALIGNMENT（动不平衡）、
    #       PAL-103（入口压力低）
    #     原先针对气蚀 / 动不平衡 / 滤网堵塞写的用例已改用密封泄漏线上的
    #     等效窗口（见 B1~B3 的 note）。若要恢复这三类覆盖，
    #     需要生成包含相应故障场景的数据集。
    # ================================================================
    DEV = "PUMP-IS100-80-160-01"

    # ---- 第一天（2026-09-13）：温度故障 → 停机 → 重启 → 恢复 ----
    T1_DAY_START   = "2026-09-13T00:00:00"
    T1_DAY_END     = "2026-09-13T23:59:55"
    T1_NORMAL_END  = "2026-09-13T00:04:55"   # 开机后的纯 NORMAL 段
    T1_DEG_START   = "2026-09-13T07:30:00"   # DEGRADING 起
    T1_WARN_START  = "2026-09-13T08:00:05"   # WARNING 起
    T1_WARN_END    = "2026-09-13T08:09:55"   # WARNING 止（随后联锁停机）
    T1_TRIP_START  = "2026-09-13T08:10:00"   # TRIP_SHUTDOWN 起
    T1_TRIP_END    = "2026-09-13T10:29:55"   # 停机段止
    T1_RESTART_END = "2026-09-13T10:55:05"   # 重启过渡止
    T1_FRAG_START  = "2026-09-13T10:55:10"   # 过渡末期的 NORMAL/WARNING 碎片

    # ---- 第二天（2026-09-14）：FAL-104 抖动 → 高温 → 停机 → 重启 → 恢复 ----
    T2_DAY_START   = "2026-09-14T00:00:00"
    T2_DAY_END     = "2026-09-14T23:59:55"
    T2_CHATTER_S   = "2026-09-14T08:34:20"   # 流量在 85 阈值附近反复穿越
    T2_CHATTER_E   = "2026-09-14T09:00:30"
    T2_WARN_START  = "2026-09-14T09:00:35"   # 高温 WARNING 起
    T2_WARN_END    = "2026-09-14T09:59:55"   # 高温 WARNING 止（随后停机）
    T2_TRIP_START  = "2026-09-14T10:00:00"
    T2_TRIP_END    = "2026-09-14T12:00:15"
    T2_RESTART_S   = "2026-09-14T12:00:20"
    T2_RESTART_END = "2026-09-14T12:25:40"

    # ================================================================
    # 测试用例（期望值全部取自新数据集的实测值）
    # ================================================================
    cases = [
        # ================= A 组：基础 =================
        ("A1", "纯 NORMAL 段（第一天开机）",
         DEV, T1_DAY_START, T1_NORMAL_END, "",
         {"points": 60, "phases": 1, "alarm_count": 0,
          "states": ["NORMAL"],
          "effective_alarm_codes": "NONE",
          "note": "单段 NORMAL，无任何告警"}),

        ("A2", "完整故障周期（DEGRADING→WARNING）",
         DEV, T1_DEG_START, T1_WARN_END, "",
         {"points": 480,
          "phases_min": 2, "phases_max": 3,
          "states_contains": ["DEGRADING", "WARNING"],
          "alarm_count_min": 4,
          "effective_alarm_codes_contains": ["TAH-101", "TAHH-101", "VAH-102", "VAHH-102"],
          "alarm_contains": ["BEARING_TEMP_DE_TRIP", "BEARING_TEMP_DE_RAMP_SHARP", "VIB_DE_TRIP"],
          "note": "温度 45→85.5℃ 越停机线，振动最高 5.12mm/s 越国标"}),

        ("A3", "只截 WARNING 段（联锁停机前 10 分钟）",
         DEV, T1_WARN_START, T1_WARN_END, "TAHH-101;VAHH-102",
         {"points": 119, "phases": 1,
          "states": ["WARNING"],
          "alarm_count_min": 4,
          "alarm_contains": ["BEARING_TEMP_DE_TRIP", "BEARING_TEMP_DE_RAMP_SHARP", "VIB_DE_TRIP"],
          "effective_alarm_codes_contains": ["TAH-101", "TAHH-101", "VAH-102", "VAHH-102"],
          "note": "最陡段温度斜率 1.53 ℃/min"}),

        ("A4", "只截 TRIP 段（停机冷却）",
         DEV, T1_TRIP_START, "2026-09-13T08:20:00", "",
         {"points": 121, "phases": 1, "alarm_count": 0,
          "states": ["TRIP_SHUTDOWN"],
          "effective_alarm_codes": "NONE",
          "note": "停机段不产生告警"}),

        # ================= B 组：故障类型与数据形态 =================
        ("B1", "密封泄漏停机前：高温越线 + 振动超标 + 流量低",
         DEV, T2_WARN_START, T2_WARN_END, "",
         {"points": 713, "phases": 1,
          "states": ["WARNING"],
          "alarm_count_min": 4,
          "alarm_contains": ["BEARING_TEMP_DE_TRIP", "VIB_DE_TRIP", "FLOW_DEV"],
          "effective_alarm_codes_contains": ["FAL-104", "TAHH-101", "VAHH-102"],
          "note": "温度最高 89.4℃、振动 4.97mm/s、流量偏离 -20%（原 B1 气蚀用例的替代）"}),

        ("B2", "流量阈值抖动（FAL-104 反复穿越）",
         DEV, T2_CHATTER_S, T2_CHATTER_E, "FAL-104",
         {"points": 315,
          "phases_min": 40, "phases_max": 70,
          "alarm_count_min": 1,
          "effective_alarm_codes_contains": ["FAL-104"],
          "note": "流量在 85m³/h 阈值附近反复穿越，切出 56 个碎片段"
                  "（原 B2 动不平衡用例的替代）★ 码化后去重只剩 1 种码"}),

        ("B3", "重启过渡：流量低 + 电机电流欠载（第一天）",
         DEV, "2026-09-13T10:30:00", T1_RESTART_END, "FAL-104;IAL-105",
         {"points": 302, "phases": 1,
          "states": ["WARNING"],
          "alarm_count_min": 3,
          "alarm_contains": ["FLOW_DEV", "PRESS_OUT_DEV", "CURRENT_LOW"],
          "effective_alarm_codes_contains": ["FAL-104", "IAL-105"],
          "note": "重启爬坡期流量偏离 -58%、出口压力只有额定的 42%、电流均值 9.45A"
                  "（原 B3 滤网堵塞用例的替代）"}),

        ("B4", "真实一天（第一天：1 次故障周期 + 1 次重启）",
         DEV, T1_DAY_START, T1_DAY_END, "",
         {"points_min": 17000, "points_max": 17500,
          "phases_min": 8, "phases_max": 14,
          "states_set_contains": ["NORMAL", "DEGRADING", "WARNING", "TRIP_SHUTDOWN"],
          "alarm_count_min": 8,
          "max_desc_len": 1000,
          "note": "故障占比约 3%，平稳段应被合并描述"}),

        ("B4S", "⚠ 高碎片密度窗口（第二天 FAL-104 抖动段）",
         DEV, T2_CHATTER_S, T2_CHATTER_E, "",
         {"points_min": 310, "points_max": 320,
          "phases_min": 40, "phases_max": 70,
          "alarm_count_min": 1,
          "effective_alarm_codes_contains": ["FAL-104"],
          "note": "56 个碎片段，验证按 operating_state 切段的粒度；不设描述长度上限"}),

        ("B5", "24h 窗口 LLM 描述长度自适应（第一天）",
         DEV, T1_DAY_START, T1_DAY_END, "",
         {"points_min": 17000, "points_max": 17500,
          "phases_min": 8, "phases_max": 14,
          "max_desc_len": 1000,
          "min_desc_len": 150,
          "note": "24h 内仅约 1 小时故障，LLM 应聚焦故障段、合并 NORMAL 段"}),

        ("E5S", "⚠ 第二天全天（碎片最多的 24h 窗口）",
         DEV, T2_DAY_START, T2_DAY_END, "",
         {"points_min": 17000, "points_max": 17500,
          "phases_min": 40, "phases_max": 80,
          "alarm_count_min": 8,
          "effective_alarm_codes_contains": ["FAL-104", "IAL-105", "TAHH-101", "VAHH-102"],
          "max_desc_len": 1000,
          "note": "63 个碎片段、16 条告警，描述应聚焦异常而非逐段罗列"}),

        # ================= C 组：时间窗口边界 =================
        ("C1", "单点窗口",
         DEV, T1_DAY_START, T1_DAY_START, "",
         {"points": 1, "phases": 1, "alarm_count": 0,
          "note": "单点窗口优雅降级，斜率=0"}),

        ("C2", "两点窗口",
         DEV, T1_DAY_START, "2026-09-13T00:00:05", "",
         {"points": 2, "phases": 1, "alarm_count": 0,
          "note": "两点窗口斜率由噪声主导，但不触发告警"}),

        ("C3", "极短窗口 1 分钟（NORMAL 段内）",
         DEV, T1_DAY_START, "2026-09-13T00:01:00", "",
         {"points": 13, "phases": 1, "alarm_count": 0,
          "note": "温度绝对值 <60℃ 时抑制斜率告警"}),

        ("C4", "停机冷却段（TRIP 内）",
         DEV, T1_TRIP_START, "2026-09-13T08:20:00", "",
         {"points": 121, "phases": 1, "alarm_count": 0,
          "note": "停机段的温度负斜率不报'急剧恶化'"}),

        ("C5", "启动过渡段（第二天重启，与 B3 形成两天对照）",
         DEV, T2_RESTART_S, T2_RESTART_END, "FAL-104;IAL-105",
         {"points": 305, "phases": 1,
          "states": ["WARNING"],
          "alarm_count_min": 3,
          "alarm_contains": ["FLOW_DEV", "PRESS_OUT_DEV", "CURRENT_LOW"],
          "note": "两天的重启过渡形态一致，可用于回归对比"}),

        ("C6", "未知设备（无数据降级）",
         "PUMP-UNKNOWN", T1_DAY_START, "2026-09-13T00:15:00", "",
         {"points": 0, "phases": 0, "alarm_count": 0,
          "quality_status": "EMPTY",
          "quality_reason_contains": ["无 SCADA 记录"],
          "note": "降级不再靠中文告警，改由 data.quality 表达；不抛异常、不调大模型"}),

        # ================= D 组：数据一致性（历史 Bug 的守护）=================
        ("D1", "✅ 数据Bug1已修（流量<85 但状态正确标 WARNING）",
         DEV, "2026-09-14T09:30:00", T2_WARN_END, "FAL-104",
         {"points": 360, "phases": 1,
          "states": ["WARNING"],
          "alarm_count_min": 1,
          "alarm_contains": ["FLOW_DEV"],
          "no_rule_hits_on_states": ["NORMAL"],
          "expect_data_bug": False,
          "note": "流量<85 且状态为 WARNING，不再出现 NORMAL 段的流量偏离误报"}),

        ("D2", "✅ 数据Bug2已修（WARNING→TRIP 有过渡，不再 5 秒突变）",
         DEV, "2026-09-13T08:09:55", "2026-09-13T08:10:10", "",
         {"points": 4, "phases": 2,
          "states": ["WARNING", "TRIP_SHUTDOWN"],
          "alarm_count_min": 1,
          "expect_data_bug": False,
          "note": "跨 WARNING→TRIP 边界正确切段"}),

        ("D3", "✅ 数据Bug3已修（启动末期正确标 WARNING）",
         DEV, "2026-09-13T10:54:55", "2026-09-13T10:55:15", "FAL-104",
         {"points": 5, "phases": 2,
          "states": ["WARNING", "NORMAL"],
          "alarm_count_min": 1,
          "alarm_contains": ["FLOW_DEV"],
          "no_rule_hits_on_states": ["NORMAL"],
          "expect_data_bug": False,
          "note": "流量<85 时状态正确标 WARNING，不误标 NORMAL"}),

        ("D4", "停机后立即重启（跨 TRIP→WARNING 边界）",
         DEV, "2026-09-13T10:29:55", "2026-09-13T10:30:30", "",
         {"points": 8, "phases": 2,
          "states": ["TRIP_SHUTDOWN", "WARNING"],
          "note": "TRIP→WARNING 边界正确切段"}),

        # ================= E 组：报警码来源（只认窗口数据） =================
        ("E1", "有报警传入 + 窗口内有报警（以窗口数据为准）",
         DEV, T1_DEG_START, T1_WARN_END, "TAHH-101;VAHH-102",
         {"points": 480,
          "phases_min": 2, "phases_max": 3,
          "effective_alarm_codes_contains": ["TAH-101", "TAHH-101", "VAH-102", "VAHH-102"],
          "note": "窗口内出现过的报警码去重排序；入参不参与计算"}),

        ("E2", "有报警传入 + 窗口内无报警（★ 不再回退到入参）",
         DEV, T1_DAY_START, T1_NORMAL_END, "TAHH-101;VAHH-102",
         {"points": 60, "phases": 1, "alarm_count": 0,
          "effective_alarm_codes": "NONE",
          "note": "★ 2026-09-17 行为变更：Step 2 不再读用户入参的 alarm_code，"
                  "窗口内没有报警就是 NONE —— 报警码只有一个真相来源（窗口数据）"}),

        ("E3", "无报警传入 + 窗口内有报警",
         DEV, T1_DEG_START, T1_WARN_END, "",
         {"points": 480,
          "phases_min": 2, "phases_max": 3,
          "effective_alarm_codes_contains": ["TAH-101", "TAHH-101", "VAH-102", "VAHH-102"],
          "note": "Step 2 从窗口数据自行提取报警码，不依赖入参"}),

        ("E4", "报警停机后清零（TRIP 段）",
         DEV, T1_TRIP_START, "2026-09-13T08:20:00", "",
         {"points": 121, "phases": 1, "alarm_count": 0,
          "effective_alarm_codes": "NONE",
          "note": "停机后 alarm_code 归 NONE"}),

        ("E5", "多故障混合报警码（第一天全天）",
         DEV, T1_DAY_START, T1_DAY_END, "",
         {"points_min": 17000, "points_max": 17500,
          "phases_min": 8, "phases_max": 14,
          "alarm_count_min": 8,
          "effective_alarm_codes_contains": [
              "TAH-101", "TAHH-101", "VAH-102", "VAHH-102", "FAL-104",
          ],
          "max_desc_len": 1000,
          "note": "一天内经历 温度故障→停机→重启欠载→恢复"}),

        # ================= G 组：状态机 =================
        ("G1", "无切换（单段 NORMAL）",
         DEV, T1_DAY_START, T1_NORMAL_END, "",
         {"points": 60, "phases": 1, "alarm_count": 0,
          "states": ["NORMAL"],
          "note": "起止状态均为 NORMAL"}),

        ("G2", "多状态切换（DEGRADING→WARNING→TRIP_SHUTDOWN）",
         DEV, T1_DEG_START, T1_TRIP_END, "",
         {"points": 2160, "phases": 3,
          "states": ["DEGRADING", "WARNING", "TRIP_SHUTDOWN"],
          "alarm_count_min": 4,
          "note": "3 段状态机演变路径正确"}),

        # ================= H 组：迟滞尾与碎片 =================
        ("H1", "过渡末期碎片（第一天重启后 7 点切 4 段）",
         DEV, T1_FRAG_START, "2026-09-13T10:55:40", "FAL-104",
         {"points": 7, "phases": 4,
          "states": ["NORMAL", "WARNING", "NORMAL", "WARNING"],
          "states_set_contains": ["NORMAL", "WARNING"],
          "alarm_count_min": 1,
          "note": "10:55:10~10:55:40 共 7 点，按 operating_state 切成 4 段"}),

        ("H2", "碎片切分粒度（第二天 FAL-104 抖动起始）",
         DEV, T2_CHATTER_S, "2026-09-14T08:39:20", "FAL-104",
         {"points": 61, "phases": 3,
          "states_set_contains": ["NORMAL", "WARNING"],
          "alarm_count_min": 1,
          "effective_alarm_codes_contains": ["FAL-104"],
          "note": "61 点被切成 3 段，验证碎片场景下的切段稳定性"}),
    ]

    # ================================================================
    # 执行
    # ================================================================
    summary = []
    for case in cases:
        case_id, name, dev, s, e, ac = case[:6]
        expect = case[6] if len(case) > 6 else None
        result, passed = _run_case(case_id, name, dev, s, e, ac, expect)
        summary.append((case_id, name, result, passed))

    # ================================================================
    # 汇总
    # ================================================================
    print(f"\n{'═' * 80}")
    print("全部用例执行完毕 - 汇总")
    print(f"{'═' * 80}")
    print(f"{'用例':<6}{'名称':<34}{'点数':>6}{'段数':>6}{'告警':>5}{'报警码':>7}{'结果':>8}")
    print(f"{'-' * 80}")

    total = 0
    passed_count = 0
    for case_id, name, result, ok in summary:
        total += 1
        if ok:
            passed_count += 1
        if result is None:
            print(f"{case_id:<6}{name:<34}{'❌ 异常':>6}")
            continue
        m = result["data"].metrics
        o = m.get("overall", {})
        codes = result["data"].alarms.effective
        n_codes = 0 if codes in ("", "NONE") else len([c for c in codes.split(";") if c.strip()])
        mark = "✅" if ok else "❌"
        print(f"{case_id:<6}{name:<34}"
              f"{o.get('total_points', 0):>6}"
              f"{o.get('phase_count', 0):>6}"
              f"{len(result['data'].threshold_flags):>5}"
              f"{n_codes:>7}"
              f"{mark:>8}")

    print(f"{'═' * 80}")
    print(f"通过: {passed_count}/{total}")
    print(f"{'═' * 80}")