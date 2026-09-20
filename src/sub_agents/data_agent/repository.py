#data/scada_model.py
"""
SCADA 时序数据 ORM 模型
=========================
表名：scada_telemetry
依据：《IS100-80-160 工业离心泵 SCADA 时序数据标准与异常诊断规范》第二节「测点数据字典」。

设计要点：
  1. 联合索引 idx_device_time (device_id, timestamp)：Step 1 传来的查询永远是
     「某设备 + 某时间窗口」，走最左前缀，避免全表扫描。
  2. 动静解耦：本表只存 device_id 与数值测点，型号等静态信息放 asset_registry。
  3. 查询一律 ORDER BY timestamp（命中联合索引，无 filesort）。
  4. 时间戳统一用 UTC naive datetime 落库，接口层负责 ISO 8601 时区解析。
"""

import enum
from datetime import datetime

from langchain_core.tools import tool
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    Index,
    String,
    func,
)

from src.utils.database import Base


class OperatingState(str, enum.Enum):
    """设备运行健康状态标签（系统标注状态机）"""

    NORMAL = "NORMAL"
    DEGRADING = "DEGRADING"
    WARNING = "WARNING"
    CRITICAL_CAVITATION = "CRITICAL_CAVITATION"
    UNBALANCE_MISALIGNMENT = "UNBALANCE_MISALIGNMENT"
    TRIP_SHUTDOWN = "TRIP_SHUTDOWN"


class ScadaTelemetry(Base):
    """SCADA 时序遥测表。一行 = 一个 5s 采样帧。"""

    __tablename__ = "scada_telemetry"

    # ---------------- 主键 ----------------
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="自增代理主键")

    # ---------------- 时间与设备 ----------------
    timestamp = Column(DateTime, nullable=False, comment="UTC 采集时间戳（5s 采样）")
    device_id = Column(String(64), nullable=False, comment="设备资产位号")

    # ---------------- 水力测点 ----------------
    flow_rate = Column(Float, nullable=False, comment="瞬时输送流量 m³/h（额定 100）")
    press_out = Column(Float, nullable=False, comment="泵出口压力 MPa（额定 0.312）")
    press_in  = Column(Float, nullable=False, comment="泵入口压力 MPa（额定 0.020）")

    # ---------------- 温度测点 ----------------
    temp_de  = Column(Float, nullable=False, comment="驱动端轴承温度 ℃（额定 45）")
    temp_nde = Column(Float, nullable=False, comment="非驱动端轴承温度 ℃（额定 45）")

    # ---------------- 振动测点 ----------------
    vib_rms_de  = Column(Float, nullable=False, comment="驱动端振动速度有效值 mm/s")
    vib_rms_nde = Column(Float, nullable=False, comment="非驱动端振动速度有效值 mm/s")

    # ---------------- 电气测点 ----------------
    motor_current = Column(Float, nullable=False, comment="电机三相运行电流 A（额定 22.5）")

    # ---------------- 状态与报警 ----------------
    operating_state = Column(
        SAEnum(OperatingState, native_enum=False, length=32, validate_strings=True),
        nullable=False,
        default=OperatingState.NORMAL,
        comment="系统标注状态机：正常/劣化/预警/气蚀/不平衡/停机",
    )
    alarm_code = Column(
        String(128), nullable=False, default="NONE",
        comment="ISA-5.1 报警代号，多码以分号连接，无报警为 NONE",
    )

    # ---------------- 审计字段 ----------------
    created_at = Column(
        DateTime, server_default=func.now(), nullable=False, comment="入库时间"
    )

    # ---------------- 索引 ----------------
    __table_args__ = (
        Index("idx_device_time", "device_id", "timestamp"),
        Index("idx_device_alarm", "device_id", "alarm_code"),
        {
            "mysql_charset": "utf8mb4",
            "mysql_engine": "InnoDB",
            "comment": "SCADA 动态时序遥测表（5s 采样，动静解耦）",
        },
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ScadaTelemetry {self.device_id} {self.timestamp:%Y-%m-%dT%H:%M:%S} "
            f"Q={self.flow_rate} Pout={self.press_out} Tde={self.temp_de} "
            f"Vde={self.vib_rms_de} state={self.operating_state.value} alarm={self.alarm_code}>"
        )

