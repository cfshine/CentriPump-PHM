"""LangGraph 感知的项目日志。

业务节点只需 ``logger.info("...")``。当调用发生在 LangGraph 节点中时，
本模块从 ``langgraph.config.get_config()`` 读取框架已经设置的运行配置；
不包装节点、不在节点中手动写 trace_id/node。
"""

from __future__ import annotations

import atexit
import copy
import logging
import queue
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"
_LOGGER_NAMESPACE = "centripump"


@dataclass(frozen=True, slots=True)
class LogContext:
    """一条日志可用的执行上下文。未知值统一为 ``-``。"""

    trace_id: str = "-"
    thread_id: str = "-"
    run_id: str = "-"
    node: str = "-"
    graph_path: str = "-"
    step: str = "-"
    node_attempt: str = "-"


_entry_context: ContextVar[LogContext] = ContextVar(
    "centripump_entry_log_context", default=LogContext()
)


def _as_text(value: Any) -> str:
    return str(value) if value not in (None, "") else "-"


def _namespace_labels(namespace: Any) -> list[str]:
    """将 LangGraph checkpoint namespace 转成可读节点层级。

    ``langgraph_checkpoint_ns`` 是 LangGraph 在节点配置中提供的运行元数据。
    每段为 ``node_name:task_id``，其中 task_id 是实现细节，不写入日志。
    此函数只作为路径增强：字段不存在或格式变化时，日志仍会正常输出。
    """

    if not isinstance(namespace, str) or not namespace:
        return []
    labels: list[str] = []
    for segment in namespace.split("|"):
        label, separator, _task_id = segment.rpartition(":")
        labels.append(label if separator and label else segment)
    return labels


def _langgraph_context() -> LogContext | None:
    """读取当前 LangGraph 节点配置；图外调用返回 ``None``。"""

    try:
        from langgraph.config import get_config

        config = get_config()
    except (ImportError, RuntimeError):
        return None

    configurable = config.get("configurable") or {}
    metadata = config.get("metadata") or {}
    if not isinstance(configurable, Mapping):
        configurable = {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    trace_id = _as_text(configurable.get("trace_id"))
    thread_id = _as_text(configurable.get("thread_id"))
    run_id = _as_text(config.get("run_id") or configurable.get("run_id"))
    node = _as_text(metadata.get("langgraph_node"))
    step = _as_text(metadata.get("langgraph_step"))
    graph_name = _as_text(configurable.get("log_graph_name"))
    labels = _namespace_labels(metadata.get("langgraph_checkpoint_ns"))

    # 顶层节点的 namespace 也包含当前 node，避免 ``diagnosis/vision/vision``。
    if labels and labels[-1] == node:
        labels.pop()
    graph_path = "/".join(part for part in (graph_name, *labels, node) if part != "-")

    node_attempt = "-"
    try:
        from langgraph.runtime import get_runtime

        execution_info = get_runtime().execution_info
        if execution_info is not None:
            run_id = _as_text(execution_info.run_id) if run_id == "-" else run_id
            thread_id = _as_text(execution_info.thread_id) if thread_id == "-" else thread_id
            node_attempt = _as_text(execution_info.node_attempt)
    except (ImportError, RuntimeError):
        pass

    return LogContext(
        trace_id=trace_id,
        thread_id=thread_id,
        run_id=run_id,
        node=node,
        graph_path=graph_path or "-",
        step=step,
        node_attempt=node_attempt,
    )


def get_log_context() -> LogContext:
    """获取当前日志上下文，优先使用 LangGraph 当前节点的上下文。"""

    return _langgraph_context() or _entry_context.get()


@contextmanager
def log_context(*, trace_id: str, thread_id: str, graph_name: str = "diagnosis") -> Iterator[None]:
    """让入口日志和图外日志也关联到同一次诊断。

    节点内日志不需要调用它：节点内会被 LangGraph 上下文自动覆盖。
    ContextVar 对线程和 asyncio task 隔离。
    """

    token = _entry_context.set(
        LogContext(trace_id=trace_id, thread_id=thread_id, graph_path=graph_name)
    )
    try:
        yield
    finally:
        _entry_context.reset(token)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = get_log_context()
        record.trace_id = context.trace_id
        record.thread_id = context.thread_id
        record.run_id = context.run_id
        record.node = context.node
        record.graph_path = context.graph_path
        record.langgraph_step = context.step
        record.node_attempt = context.node_attempt
        return True


class _PreservingQueueHandler(QueueHandler):
    """QueueHandler 默认会丢弃 exc_info；这里保留已格式化的 traceback。"""

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        prepared = copy.copy(record)
        prepared.message = prepared.getMessage()
        prepared.msg = prepared.message
        prepared.args = None
        if prepared.exc_info:
            prepared.exc_text = logging.Formatter().formatException(prepared.exc_info)
            prepared.exc_info = None
        return prepared


class _PrettyFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        trace_id = record.trace_id[:12] if record.trace_id != "-" else "-"
        level = f"{record.levelname:<8}"
        color = self.COLORS.get(record.levelno, "")
        result = (
            f"{timestamp} │ {color}{level}{self.RESET} │ "
            f"{record.graph_path:<36} │ [{trace_id}] │ {record.getMessage()}"
        )
        if getattr(record, "exc_text", None):
            result += "\n" + record.exc_text
        return result


class _FileFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s │ %(levelname)-8s │ graph=%(graph_path)s │ "
            "node=%(node)s │ trace=%(trace_id)s │ thread=%(thread_id)s │ "
            "run=%(run_id)s │ step=%(langgraph_step)s │ attempt=%(node_attempt)s │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


_listener: QueueListener | None = None
_configured = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """初始化一次进程内日志输出（终端 + 10 MiB 滚动文件）。"""

    global _configured, _listener
    logger = logging.getLogger(_LOGGER_NAMESPACE)
    if _configured:
        logger.setLevel(level)
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(level)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(_PrettyFormatter())

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(_FileFormatter())

    log_queue: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
    queue_handler = _PreservingQueueHandler(log_queue)
    queue_handler.addFilter(_ContextFilter())
    logger.addHandler(queue_handler)

    _listener = QueueListener(log_queue, console, file_handler, respect_handler_level=True)
    _listener.start()
    atexit.register(shutdown_logging)
    _configured = True
    return logger


def shutdown_logging() -> None:
    """停止后台日志队列；通常由进程退出钩子自动调用。"""

    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None


def get_logger(name: str) -> logging.Logger:
    """返回项目命名空间内的 Logger。"""

    setup_logging()
    suffix = name.removeprefix("src.")
    return logging.getLogger(f"{_LOGGER_NAMESPACE}.{suffix}")


def build_graph_config(
    *, trace_id: str, thread_id: str | None = None, graph_name: str = "diagnosis",
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构造传给 ``invoke`` / ``ainvoke`` 的可继承 RunnableConfig。"""

    result = dict(config or {})
    configurable = dict(result.get("configurable") or {})
    configurable.update(
        {"trace_id": trace_id, "thread_id": thread_id or trace_id, "log_graph_name": graph_name}
    )
    result["configurable"] = configurable
    return result


def invoke_graph(graph: Any, state: Any, *, trace_id: str, thread_id: str | None = None,
                 graph_name: str = "diagnosis", config: Mapping[str, Any] | None = None) -> Any:
    """同步入口：一次调用完成 trace/thread 注入与图外日志关联。"""

    final_thread_id = thread_id or trace_id
    graph_config = build_graph_config(
        trace_id=trace_id, thread_id=final_thread_id, graph_name=graph_name, config=config
    )
    with log_context(trace_id=trace_id, thread_id=final_thread_id, graph_name=graph_name):
        return graph.invoke(state, config=graph_config)


async def ainvoke_graph(graph: Any, state: Any, *, trace_id: str, thread_id: str | None = None,
                        graph_name: str = "diagnosis", config: Mapping[str, Any] | None = None) -> Any:
    """异步入口；ContextVar 会随当前 asyncio task 传播。"""

    final_thread_id = thread_id or trace_id
    graph_config = build_graph_config(
        trace_id=trace_id, thread_id=final_thread_id, graph_name=graph_name, config=config
    )
    with log_context(trace_id=trace_id, thread_id=final_thread_id, graph_name=graph_name):
        return await graph.ainvoke(state, config=graph_config)
