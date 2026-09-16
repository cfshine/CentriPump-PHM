#scripts/export_csv.py
"""
SCADA 数据导出为 CSV。
用法：
    python -m scripts.export_csv                          # 默认：按天分文件
    python -m scripts.export_csv --mode full              # 全量单文件
    python -m scripts.export_csv --mode by_state          # 按状态分文件
    python -m scripts.export_csv --out-dir ./exports      # 指定输出目录
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.utils.database import engine    # ← 按框架实际路径调整
# 若仍在独立开发：from core.database import engine


DEFAULT_DEVICE = "PUMP-IS100-80-160-01"
DEFAULT_OUT_DIR = "./exports"


# ==================== 通用查询 ====================

def _fetch_df(device_id: str) -> pd.DataFrame:
    """拉取全量数据到 DataFrame。"""
    sql = text("""
        SELECT
            timestamp,
            device_id,
            flow_rate,
            press_out,
            press_in,
            temp_de,
            temp_nde,
            vib_rms_de,
            vib_rms_nde,
            motor_current,
            operating_state,
            alarm_code
        FROM scada_telemetry
        WHERE device_id = :dev
        ORDER BY timestamp ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"dev": device_id}).fetchall()
        columns = [
            "timestamp", "device_id", "flow_rate", "press_out", "press_in",
            "temp_de", "temp_nde", "vib_rms_de", "vib_rms_nde",
            "motor_current", "operating_state", "alarm_code",
        ]
        df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ==================== 导出模式 ====================

def export_by_day(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """按天分文件导出。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for day, group in df.groupby(df["timestamp"].dt.date):
        # 统计该天的故障类型，用于文件名
        alarms = group[group["alarm_code"] != "NONE"]["alarm_code"]
        if alarms.empty:
            suffix = "normal"
        else:
            # 提取该天最"高级"的报警码作为标识
            alarm_set = set()
            for codes in alarms:
                alarm_set.update(codes.split(";"))   
            if "VAHH-102" in alarm_set or "TAHH-101" in alarm_set:
                suffix = "seal_leak" if "TAHH-101" in alarm_set else "fault"
            elif "VAH-102" in alarm_set or "TAH-101" in alarm_set:
                suffix = "warning"
            else:
                suffix = "abnormal"
        
        filename = f"day_{day.isoformat()}_{suffix}.csv"
        filepath = out_dir / filename
        group.to_csv(filepath, index=False, encoding="utf-8-sig")
        files.append(filepath)
        print(f"  ✅ {filepath}  ({len(group)} 行)")
    return files


def export_full(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """全量单文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"scada_full_{df['timestamp'].min().date()}_to_{df['timestamp'].max().date()}.csv"
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"  ✅ {filepath}  ({len(df)} 行)")
    return [filepath]


def export_by_state(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """按 operating_state 分文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for state, group in df.groupby("operating_state"):
        filename = f"state_{state}.csv"
        filepath = out_dir / filename
        group.to_csv(filepath, index=False, encoding="utf-8-sig")
        files.append(filepath)
        print(f"  ✅ {filepath}  ({len(group)} 行)")
    return files


# ==================== CLI ====================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 SCADA 数据为 CSV")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help=f"设备 ID（默认 {DEFAULT_DEVICE}）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"输出目录（默认 {DEFAULT_OUT_DIR}）")
    parser.add_argument("--mode", choices=["by_day", "full", "by_state"],
                        default="by_day",
                        help="导出模式（默认 by_day）")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)

    print(f"[Export] 设备={args.device}")
    print(f"[Export] 模式={args.mode}")
    print(f"[Export] 输出目录={out_dir.resolve()}")
    print()

    df = _fetch_df(args.device)
    if df.empty:
        print(f"❌ 设备 {args.device} 无数据")
        return 1

    print(f"[Export] 读到 {len(df)} 行，"
          f"时间跨度 {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print()

    if args.mode == "by_day":
        files = export_by_day(df, out_dir)
    elif args.mode == "full":
        files = export_full(df, out_dir)
    else:
        files = export_by_state(df, out_dir)

    print()
    print(f"[Export] 完成，共 {len(files)} 个文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())