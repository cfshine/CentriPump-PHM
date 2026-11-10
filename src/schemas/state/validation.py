"""节点输出契约与 checkpoint 序列化校验。

主防线不再是模型构造时的手写类型扫描，而是：

1. **节点边界白名单**：每个节点只能返回自己拥有的 State 顶层键；
2. **真实 serde 往返**：用项目实际使用的 ``SqliteSaver.serde`` 对更新值做
   序列化 → 反序列化，并比较往返前后是否一致。

这两项都在节点返回时执行，因此任何违反契约的节点会在运行现场立即失败，
而不是等到 checkpoint 落盘时才报一个难以定位的序列化错误。
"""

from __future__ import annotations

import sqlite3
from functools import wraps
from typing import Any, Callable, TypeVar

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel


#: 每个节点允许返回的 State 顶层键。
#: 新增节点时必须在这里登记；节点返回白名单外的键会直接抛错。
NODE_OUTPUT_WHITELIST: dict[str, frozenset[str]] = {
    # 主图
    "start_node": frozenset({"context"}),
    "engineer_text": frozenset({"engineer_text"}),
    "data": frozenset({"data"}),
    "vision": frozenset({"vision"}),
    "evidence_merge": frozenset(),
    "manual": frozenset({"manual"}),
    "reasoner": frozenset({"reasoning"}),
    "safety_guard": frozenset({"safety", "manual"}),
    "reporter": frozenset({"delivery"}),
    # Step 2 子图
    "analyze": frozenset({"data"}),
    "summarize": frozenset({"data"}),
}


_serde: JsonPlusSerializer | None = None


def get_checkpoint_serde() -> JsonPlusSerializer:
    """返回与主图 checkpointer 同款的 serde 单例。"""

    global _serde
    if _serde is None:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        _serde = SqliteSaver(connection).serde
    return _serde


def assert_serde_roundtrip(value: Any, *, path: str) -> None:
    """用真实 checkpoint serde 做一次往返校验。"""

    payload = value.model_dump() if isinstance(value, BaseModel) else value
    serde = get_checkpoint_serde()
    type_str, data = serde.dumps_typed(payload)
    restored = serde.loads_typed((type_str, data))
    if restored != payload:
        raise TypeError(
            f"节点输出 {path} 无法通过 checkpoint serde 往返："
            f"序列化前与反序列化后不一致。"
        )


def validate_node_update(node_name: str, update: dict[str, Any]) -> None:
    """校验一个节点的返回值：键在白名单内，且每个值可被 checkpoint serde 往返。"""

    if node_name not in NODE_OUTPUT_WHITELIST:
        raise KeyError(
            f"节点 {node_name!r} 未登记输出白名单；"
            f"请在 NODE_OUTPUT_WHITELIST 中声明它允许写的 State 键。"
        )
    if not isinstance(update, dict):
        raise TypeError(f"节点 {node_name} 必须返回 dict，实际为 {type(update).__name__}。")

    allowed = NODE_OUTPUT_WHITELIST[node_name]
    illegal = set(update) - allowed
    if illegal:
        raise ValueError(
            f"节点 {node_name} 越权写入 {sorted(illegal)}；"
            f"允许的键为 {sorted(allowed)}。"
        )

    for key, value in update.items():
        assert_serde_roundtrip(value, path=f"{node_name}.{key}")


F = TypeVar("F", bound=Callable[..., dict[str, Any]])


def node_output(node_name: str) -> Callable[[F], F]:
    """包装节点函数，在返回处强制执行输出白名单与 serde 往返校验。"""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            update = func(*args, **kwargs)
            validate_node_update(node_name, update)
            return update

        return wrapper  # type: ignore[return-value]

    return decorator


__all__ = [
    "NODE_OUTPUT_WHITELIST",
    "assert_serde_roundtrip",
    "get_checkpoint_serde",
    "node_output",
    "validate_node_update",
]
