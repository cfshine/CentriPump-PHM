"""主控全局上下文（LangGraph 全局 State 契约）。

这是**跨 Step 的公共契约**：所有 Step 共享的字段都在这里定义，全程只定义一次。
各子 Agent 的私有状态用**继承**本类的方式扩展（见
`src/sub_agents/data_agent/state.py` 的 `DataAgentState`），不重复声明公共字段。

分层原则（实测确认过的 LangGraph 行为）：
    子图能读到的键 = 子图 schema 里声明了的键
    能回流父图的键 = 父图 schema 里也有的键
    子图独有的键   = 私有，不会回流父图

所以子图必须"继承公共 + 再补私有"；只定义私有字段的话，子图连 device_id 都读不到。

—— 当前版本的字段范围 ——
本文件目前只含 **Step 1 输入 + Step 2 产出** 中下游需要读的部分。
Step 3~7 的公共字段在各自接入时往这里加。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DiagnosisState(BaseModel):
    """全流程共享状态。

    ★ 两个关键设计：

    1. **所有字段都必须有默认值**。LangGraph 会用「部分状态」反复重建这个模型
       （每个节点只回写自己产出的那几个键），没有默认值的字段会直接校验失败。

    2. **`extra="forbid"`**。节点若回写未声明的键（字段名拼错、临时变量忘了删），
       会立刻报错，而不是静默丢数据 —— 这正是 Pydantic 相对 TypedDict 的价值。
    """

    model_config = ConfigDict(extra="forbid")

    # ===================== A 类：Step 1 写入的输入 =====================
    device_id: str = ""
    start_time: str = ""
    end_time: str = ""
    alarm_code: str = ""          # 可选，可能为空字符串 ""

    # ===================== B 类：Step 2 产出（下游 Step4/5/6/7 读取）=====
    calculated_metrics: dict[str, Any] = Field(default_factory=dict)
    threshold_flags: list[str] = Field(default_factory=list)
    effective_alarm_codes: str = "NONE"      # 从窗口数据提取的有效报警码
    last_alarm_codes: str = "NONE"           # 窗口内最后一次非 NONE 的报警码组合
    all_alarm_codes_in_window: list[str] = Field(default_factory=list)
    llm_description: str = ""                # 工况自然语言描述
    basic_judgment: str = ""                 # 基础判断说明（不涉及故障归因）
    rag_search_queries: list[str] = Field(default_factory=list)   # 供 Step 4 使用的检索词


    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in type(self).model_fields:
            raise KeyError(f"{key!r} 不在状态契约里（请先在 schemas/state.py 声明）")
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:  # noqa: A003
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in type(self).model_fields
