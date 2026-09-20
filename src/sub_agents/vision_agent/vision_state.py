"""Step 3 子图的状态：**继承公共契约，不加私有字段**。
"""

from __future__ import annotations
from src.schemas.state import DiagnosisState


class VisionAgentState(DiagnosisState):
    """Step 3 子图状态 = 公共契约（继承），当前无私有字段。

    读到的输入：``image_refs``（图片路径/URL 列表）、``device_id``、``alarm_code``。
    写回的输出：``visual_description``（人读的整段文本）、``visual_findings``（机器读的盒子）。
    """
