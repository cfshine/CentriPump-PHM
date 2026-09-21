# 项目日志使用说明

## 目标

日志系统让业务节点只关心业务，不手动携带执行信息：

```python
logger.info("开始视觉分析")
```

运行在 LangGraph 节点中时，日志会自动补齐当前诊断任务的 `trace_id`、
`thread_id`、节点名和执行层级。例如：

```text
2026-09-20 21:08:24 │ INFO │ diagnosis/vision │ [a81c29d4e2f0] │ 开始视觉分析
```

文件日志还会记录完整的 `trace_id`、`thread_id`、`run_id`、LangGraph step 和
节点重试次数，便于按一次诊断任务检索。

## 日志如何自动获得上下文

日志不是通过“猜调用栈”或给每个节点包一层 wrapper 实现的。

LangGraph 在执行每个节点时，已经把当前运行配置放在自己的上下文中。`src.core.logger`
在每次真正写日志时读取这份配置，并把字段加入当前 `LogRecord`：

| 字段                  | 来源                                            | 是否需要业务节点处理 |
| --------------------- | ----------------------------------------------- | -------------------- |
| `node`              | LangGraph 自动注入的`metadata.langgraph_node` | 不需要               |
| `step`              | LangGraph 自动注入的`metadata.langgraph_step` | 不需要               |
| `thread_id`         | 入口传入`configurable.thread_id`              | 入口一次             |
| `trace_id`          | 入口传入`configurable.trace_id`               | 入口一次             |
| `run_id` / 重试次数 | `Runtime.execution_info`（可用时）            | 不需要               |
| `graph_path`        | 入口图名 + LangGraph checkpoint namespace       | 不需要               |

这意味着节点不必接收 `RunnableConfig` 或 `Runtime` 参数，也不必调用
`set_node()`、`set_trace_id()`。

### 子图和嵌套子图

入口把根图名设置为 `diagnosis`。LangGraph 在子图运行时会携带 checkpoint
namespace；日志模块会去掉其中不适合展示的 task ID，保留节点层级。因此未来将
`data` 替换为 Data 子图后，子图内部可显示为：

```text
diagnosis/data/fetch_data
diagnosis/data/calculate_metrics
diagnosis/data/semanticize
```

`langgraph_checkpoint_ns` 是 LangGraph 的运行元数据而不是为展示设计的稳定业务
字段。因此它只用于增强展示：若未来框架未提供该字段或格式改变，日志仍可正常
输出 `node` 和 `trace_id`，只是层级会退化。

## 日常使用

### 1. 在节点模块取得 logger

```python
from src.core.logger import get_logger

logger = get_logger(__name__)
```

每个模块只创建一次。不要在函数内重复创建，也不要直接使用 `print()`。

### 2. 在节点中正常记录日志

```python
def vision_node(state: DiagnosisState) -> dict:
    logger.info("开始视觉分析")
    try:
        findings = analyze_images(state.context.image_refs)
    except Exception:
        logger.exception("视觉分析失败")
        raise
    logger.info("视觉分析完成，发现 %s 项问题", len(findings))
    return {"vision": {"findings": findings}}
```

建议：

- 正常的阶段开始、结束和降级使用 `info`。
- 输入不完整但可继续处理使用 `warning`。
- 捕获后仍要抛出的异常使用 `logger.exception(...)`，它会自动保留 traceback。
- 不要在日志正文中重复写 `[trace_id=...]` 或节点名；格式器已经会输出。

### 3. 从 API / CLI 入口启动一次诊断

`trace_id` 是业务诊断任务 ID，由入口生成；当前项目约定 `thread_id == trace_id`，
让 checkpoint、日志、报告和外部数据引用可以直接关联。

```python
from uuid import uuid4

from src.orchestrator.runner import run_diagnosis
from src.schemas.state import create_initial_state

trace_id = uuid4().hex
state = create_initial_state(
    trace_id=trace_id,
    device_id="PUMP-001",
    start_time="2026-09-20T08:00:00",
    end_time="2026-09-20T10:00:00",
    user_query="振动异常",
)

result = run_diagnosis(state)
```

`run_diagnosis()` 会统一完成：

1. 校验入口 `trace_id` 与 State 中的值一致；
2. 向 LangGraph config 注入 `trace_id`、`thread_id` 和根图名；
3. 让图开始前、结束后或失败时的入口日志也带上同一个 trace；
4. 调用 `diagnosis_graph.invoke(...)`。

因此 API 层不应再调用旧的 `set_trace_id()`。

### 4. 自行调用图（仅用于特殊场景）

通常应使用 `run_diagnosis()`。如果确实需要直接 `.invoke()`，必须使用
`build_graph_config()`，否则节点仍能显示名称，但不会有业务 `trace_id`：

```python
from src.core.logger import build_graph_config
from src.orchestrator.graph import diagnosis_graph

result = diagnosis_graph.invoke(
    state,
    config=build_graph_config(
        trace_id=trace_id,
        thread_id=trace_id,
        graph_name="diagnosis",
    ),
)
```

异步图使用 `await ainvoke_graph(...)`；它同样会维持当前 asyncio task 的日志上下文。

## 输出位置与并发行为

- 终端：彩色、简洁，默认展示短 `trace_id`。
- 文件：`logs/app.log`，保留完整检索字段。
- 滚动：单个文件 10 MiB，保留 5 个历史文件。
- 进程内并发：先写入队列，再由一个后台监听器顺序写入终端和文件；Python 的
  `ContextVar` 会隔离不同线程和 asyncio task 的入口上下文。

当前实现面向单个 Python 进程。若以后部署为多个独立 worker 进程同时写同一个
`logs/app.log`，应改为每个进程单独文件，或接入集中式日志服务；标准
`RotatingFileHandler` 不提供多进程安全滚动。

## 本次文件说明

| 文件                              | 作用                                                                                                                            |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `src/core/logger.py`            | 日志主实现：读取 LangGraph 上下文、格式化、队列输出、文件滚动，以及图调用辅助函数。                                             |
| `src/core/__init__.py`          | `core` 基础设施包标记。                                                                                                       |
| `src/orchestrator/runner.py`    | 诊断图的推荐入口；统一注入 trace/thread ID 并记录任务开始、结束、失败。                                                         |
| `src/orchestrator/nodes.py`     | 示范节点如何只使用`logger.info(...)`；不包含日志上下文管理代码。                                                              |
| `src/orchestrator/graph.py`     | 构建图前初始化日志系统；不注册任何节点 wrapper。                                                                                |
| `src/utils/logger.py`           | 旧导入路径的兼容层。已有`from src.utils.logger import get_logger` 的代码无需立即修改。新代码优先从 `src.core.logger` 导入。 |
| `tests/test_logging_context.py` | 验证在未包装节点的情况下，LangGraph 节点可自动取得 trace、thread、node 和 graph path。                                          |

## 排查清单

如果日志中 `node` 为 `-`，说明该日志不在 LangGraph 节点执行期间产生；这是入口或
普通工具函数的正常表现。

如果 `trace_id` 为 `-`，说明调用图时没有使用 `run_diagnosis()` 或
`build_graph_config()`。检查入口是否生成 trace 并写入初始 State。

如果日志路径只有 `diagnosis/节点名`，而没有更深子图层级，先确认子图是否以
LangGraph 编译图方式挂载和执行；普通 Python 函数调用没有 LangGraph 子图运行
上下文。
