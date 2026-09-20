"""Step 2 子图的状态：继承公共契约 + 补私有字段。

★ 这就是「父子图共享状态」的正确做法：

    父图 DiagnosisState      只定义公共字段（Step1 输入 + Step2 产出中下游要读的）
    子图 DataAgentState      继承它 → 公共字段**一处定义**，子图不重复声明
                             再补自己的私有字段 → **不污染主控 state**

实测确认的 LangGraph 行为：
    · 子图能读到的键 = 子图 schema 里声明了的键
      （所以必须继承；只定义私有字段的话，子图连 device_id 都读不到）
    · 能回流父图的键 = 父图 schema 里也有的键
      （所以下面这两个私有字段**不会**出现在主控 state 里）

★ 但要注意「私有 ≠ 免费」：
    私有字段虽然不外流到父图，**仍然留在子图自己的 state 里**，
    子图内部每走一步照样要被合并一次。24h 窗口的 raw_telemetry_data 有
    1.7 万条，这个开销是实打实的。要彻底消除，只能让它**根本不进 state**
    （把取数与计算合并成一个节点，数据走函数局部变量）——那是后续优化项，
    本轮迁移保持现有 3 节点拓扑不变。
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, SkipValidation

from src.schemas.state import DiagnosisState


class DataAgentState(DiagnosisState):
    """Step 2 子图状态 = 公共契约（继承） + 私有工作变量。"""

    # ===================== Step 2 私有（不外流主控）=====================

    #: 窗口内的原始遥测。24h 窗口可达 1.7 万条，逐条校验既无意义又拖慢
    #: 每一次状态合并，故跳过校验（SkipValidation），只保留字段名的强约束。
    raw_telemetry_data: SkipValidation[list[dict[str, Any]]] = Field(default_factory=list)

    #: 报警事件序列（语义化节点的输入材料），同样可达数百条。
    alarm_events: SkipValidation[list[dict[str, Any]]] = Field(default_factory=list)
