"""自制离线案例：原始 CSV → 主图/数据子图 → 本地规程 → 安全裁决 → 报告。

从项目根运行：python scripts/run_demo.py
不需要数据库、API 密钥或下载模型；不预填任何分析结果。
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def create_case() -> tuple[Path, Path]:
    csv_path = ROOT / "data/sample_inputs/demo_bearing.csv"
    manual_path = ROOT / "data/manuals/demo_bearing_manual.json"
    start = datetime(2026, 11, 17, 0, 0, 0)
    fields = ["device_id", "timestamp", "flow_rate", "press_out", "press_in",
              "temp_de", "temp_nde", "vib_rms_de", "vib_rms_nde", "motor_current",
              "operating_state", "alarm_code"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(361):
            minute = i / 12
            temp = 48 + max(0, minute - 5) * 1.5
            vib = 1.5 + max(0, minute - 5) * 0.15
            writer.writerow(dict(device_id="PUMP-DEMO-BEARING-01",
                timestamp=(start + timedelta(seconds=i * 5)).isoformat(),
                flow_rate=round(100 + math.sin(i / 10) * 0.5, 3),
                press_out=0.312, press_in=0.02, temp_de=round(temp, 3),
                temp_nde=45, vib_rms_de=round(vib, 3), vib_rms_nde=1.4,
                motor_current=22.5,
                operating_state="NORMAL" if minute < 5 else "DEGRADING",
                alarm_code="TAHH-101" if temp >= 80 else ""))
    manual_path.write_text(json.dumps({
        "title": "自制轴承故障演示规程（仅用于集成测试）",
        "clauses": [{
            "doc_id": "DEMO-BEARING-MANUAL", "chunk_id": "bearing-01",
            "section": "1. 驱动端轴承温振异常（演示资料）",
            "rule_codes": ["BEARING_TEMP_DE_WARN", "BEARING_TEMP_DE_TRIP", "VIB_DE_WARN", "VIB_DE_TRIP"],
            "keywords": ["轴承"],
            "quote": "演示案例采用项目 rules/thresholds.py 的判据：驱动端轴承温度达到80℃或振动达到4.5mm/s触发停机线。温振同步恶化时应优先核验轴承及润滑系统，具体根因须检查确认。",
            "requirement": "达到停机线时应按现场审批规程隔离设备，禁止带故障继续运行。",
            "action": "隔离、冷却并确认安全后检查润滑状态、轴承游隙及磨损；复测温振后由责任工程师决定是否恢复运行。",
            "mechanism": "轴承与润滑异常可伴随驱动端温度、振动同步升高；仅凭时序数据无法确认具体损伤形态。"
        }, {
            "doc_id": "DEMO-BEARING-MANUAL", "chunk_id": "cavitation-01",
            "section": "2. 气蚀对照条款（演示资料）",
            "rule_codes": ["PRESS_IN_TRIP", "CV_UNSTABLE"], "keywords": ["气蚀"],
            "quote": "演示气蚀条款：入口压力降低或流量显著波动时应核验吸入条件。"
        }]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, manual_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "docs/demo_diagnosis_report.md"))
    args = parser.parse_args()
    os.environ.setdefault("PHM_CHECKPOINT_DB", str(ROOT / "data/demo_checkpoints.db"))
    from src.orchestrator.runner import run_diagnosis
    from src.schemas.state import DiagnosisState, create_initial_state

    csv_path, manual_path = create_case()
    state = create_initial_state(
        trace_id=f"demo-bearing-{uuid4().hex[:12]}", device_id="PUMP-DEMO-BEARING-01",
        start_time="2026-11-17T00:00:00", end_time="2026-11-17T00:30:00",
        user_query="合成演示：驱动端轴承逐渐发热并伴随振动升高，请排查。未提供照片。",
    )
    state = state.model_copy(update={"context": state.context.model_copy(update={
        "analysis_mode": "offline", "telemetry_uri": str(csv_path), "manual_uri": str(manual_path),
    })})
    final = DiagnosisState.model_validate(run_diagnosis(state, report_path=args.output))
    checks = {
        "时序计算成功": final.data.quality.status == "OK" and final.data.quality.total_points == 361,
        "命中温度和振动停机线": {"BEARING_TEMP_DE_TRIP", "VIB_DE_TRIP"}.issubset(final.data.threshold_flags),
        "检索命中实际演示条款": [e.chunk_id for e in final.manual.evidences] == ["bearing-01"],
        "未伪造视觉证据": final.vision.status == "NO_IMAGE" and not final.vision.findings,
        "安全门禁拦截": final.safety.decision == "BLOCK" and final.safety.risk_level == "CRITICAL",
        "生成交付证据链": bool(final.delivery.report_sha256 and final.delivery.traceability),
    }
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    if not all(checks.values()):
        raise RuntimeError("案例验收未通过；请检查报告及日志")
    print(f"报告：{Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
