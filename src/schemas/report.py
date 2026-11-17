"""把最终诊断状态导出为可读报告和完整 JSON 证据档案。"""

import hashlib
from pathlib import Path

from src.schemas.state import DiagnosisState


def write_diagnosis_report(state: DiagnosisState, output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if state.safety.verdict is None:
        raise ValueError("缺少安全裁决，不能生成最终报告")
    ctx = state.context
    verdict = state.safety.verdict
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["# 离心泵诊断报告", "", f"- 诊断编号：`{ctx.trace_id}`",
             f"- 设备：`{ctx.device_id}`",
             f"- 数据窗口：{ctx.start_time} ～ {ctx.end_time}（案例时间为 UTC）",
             f"- 报告生成时间：{state.delivery.diagnosed_at}（运行环境本地时间）",
             f"- 运行模式：`{ctx.analysis_mode}`", "",
             "## 结论", "", f"安全裁决：**{verdict.decision.value}**；风险等级：**{verdict.risk_level.value}**。",
             f"故障模式：`{verdict.failure_mode.value}`。", "",
             "故障模式是基于现有证据的候选判断，润滑失效、疲劳损伤等具体根因仍须停机检查确认。", "",
             "## 输入与执行情况", "", f"工程师描述：{ctx.user_query}", "",
             "| 环节 | 结果 |", "|---|---|",
             f"| 工程师文本 | {state.engineer_text.status} |",
             f"| 时序分析 | {state.data.quality.status}，{state.data.quality.total_points} 点；{cell(state.data.quality.reason)} |",
             f"| 视觉分析 | {state.vision.status}；{cell(state.vision.summary)} |",
             f"| 规程检索 | {state.manual.status}，{len(state.manual.evidences)} 条 |",
             f"| 综合归因 | {cell(state.reasoning.summary)} |", ""]
    if ctx.analysis_mode == "offline":
        lines += ["本次为合成案例的离线集成验证。时序统计、规则判定和安全门禁真实执行；"
                  "数据描述采用确定性模板，归因采用现有规则。未调用语言模型、视觉模型、MySQL 或向量检索。"
                  "未提供现场照片，不构造视觉证据。演示规程是自制测试资料，不能替代正式设备手册或经审核的生产规程。", ""]
    lines += ["## 窗口统计", "", "| 指标 | 数值 | 单位 |", "|---|---:|---|"]
    for m in state.data.measurements:
        if m.segment == "overall":
            lines.append(f"| {cell(m.name)} | {m.value:g} | {cell(m.unit)} |")
    lines += ["", "## 分段与规则", "", "| 阶段 | 开始 | 结束 | 点数 | 命中规则 |", "|---|---|---|---:|---|"]
    for seg in state.data.segments:
        lines.append(f"| {seg.state} | {seg.start_time} | {seg.end_time} | {seg.data_points} | {cell(', '.join(seg.rule_hits))} |")
    lines += ["", "安全门禁判定理由：", ""] + [f"- {reason}" for reason in verdict.reasons]
    lines += ["", "## 规程证据", ""]
    for ev in state.manual.evidences:
        lines += [f"### {ev.doc_id} / {ev.chunk_id}", "",
                  f"位置：{ev.section}；检索方式：{ev.channel}；匹配词覆盖率：{ev.score:.2f}。", "", ev.quote, ""]
    if not state.manual.evidences:
        lines += ["未检索到规程证据。", ""]
    lines += ["## 安全处置与后续核验", ""]
    lines += [f"{i}. [{a.priority}] {a.action}（依据：`{a.source_ref.id}`）"
              for i, a in enumerate(state.manual.actions, 1)]
    lines += ["", "上述为诊断建议；本次运行未连接控制系统，也未实际停机或下发工单。", "",
              "## 溯源与完整性", "", f"- 规则版本：`{verdict.rule_version}`",
              f"- 门禁输入 SHA-256：`{verdict.input_hash}`",
              f"- 裁决与证据链 SHA-256：`{state.delivery.report_sha256}`（不代表整份 Markdown 文件哈希）",
              f"- 逻辑交付地址：`{state.delivery.report_uri}`",
              f"- 完整状态档案：`{path.with_suffix('.json').name}`", ""]
    for source in [ctx.telemetry_uri, ctx.manual_uri]:
        if source:
            source_path = Path(source)
            lines.append(f"- 输入文件：`{source_path}`；SHA-256：`{hashlib.sha256(source_path.read_bytes()).hexdigest()}`")
    lines += [""] + [f"- {r.kind.value}: `{r.id}` — {r.locator}" for r in state.delivery.traceability]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return path
