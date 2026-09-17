from __future__ import annotations

import os
import time
import uuid
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool

from src.utils.logger import logger


T = TypeVar("T", bound=BaseModel)


# ================================================================
# Pydantic Model → Tool Schema
# ================================================================

def pydantic_to_tool(
    schema: Type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> dict[str, Any]:
    """
    将 Pydantic Model 转换成 OpenAI / LangChain Tool Schema。

    注意：
    这里创建的只是“工具描述”，不是一个真正执行的 Tool。

    它的作用是：

        Pydantic Model
              ↓
        JSON Schema
              ↓
        Tool Calling
              ↓
        tool_call.args
              ↓
        Pydantic Model
    """

    if not issubclass(schema, BaseModel):
        raise TypeError(
            "schema 必须是 Pydantic BaseModel 子类"
        )

    tool_name = name or schema.__name__

    tool_description = (
        description
        or schema.__doc__
        or f"返回 {tool_name} 结构化结果"
    )

    parameters = schema.model_json_schema()

    # Pydantic JSON Schema 中可能存在 title
    # Tool parameters 不需要它，删掉可以让 Schema 更干净
    parameters.pop("title", None)

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": parameters,
        },
    }


# ================================================================
# LLM Agent
# ================================================================

class LLMAgent:
    """
    LLM 统一调用网关。

    所有业务 Agent 都通过本类调用模型。

    支持：

        llm.invoke(messages)

        llm.with_tools([...])

        llm.with_struct(Result)

        llm.with_tools([...]).with_struct(Result)

    Structured Output 不使用：

        with_structured_output()

    而是使用：

        Pydantic
            ↓
        Tool Schema
            ↓
        bind_tools()
            ↓
        AIMessage.tool_calls
            ↓
        Pydantic.model_validate()
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_provider: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 30,
        max_retries: int = 2,

        trace_id: Optional[str] = None,

        # 内部状态
        _llm: Optional[BaseChatModel] = None,
        _tools: Optional[list[Any]] = None,
        _struct_schema: Optional[Type[BaseModel]] = None,
        _struct_tool_name: Optional[str] = None,
        _output_instruction: Optional[str] = None,
    ):
        self.model_name = (
            model_name
            or os.getenv("LLM_MODEL_NAME", "gpt-4o")
        )

        self.model_provider = (
            model_provider
            or os.getenv("LLM_PROVIDER", "openai")
        )

        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        self.trace_id = trace_id or uuid.uuid4().hex

        # 原始 BaseChatModel
        self._llm = _llm or self._build_model()

        # 普通工具
        self._tools = list(_tools or [])

        # Pydantic Structured Output
        self._struct_schema = _struct_schema

        # output tool 名称
        self._struct_tool_name = _struct_tool_name

        # 附加 System Prompt
        self._output_instruction = _output_instruction

    # ============================================================
    # 创建 LLM
    # ============================================================

    def _build_model(self) -> BaseChatModel:

        try:

            return init_chat_model(
                model=self.model_name,
                model_provider=self.model_provider,
                temperature=self.temperature,
                timeout=self.timeout,

                # ------------------------------------------------
                # 重试由 LLMAgent 自己控制
                # 避免双重 retry
                # ------------------------------------------------

                max_retries=0,
            )

        except Exception:

            logger.exception(
                "LLM 初始化失败 | model=%s | trace_id=%s",
                self.model_name,
                self.trace_id,
            )

            raise

    # ============================================================
    # Tool
    # ============================================================

    def with_tools(
        self,
        tools: list[Any],
    ) -> "LLMAgent":
        """
        装载 LangChain Tools。

        tools 可以是：

            @tool
            def xxx(...):
                ...

        也可以是其他 LangChain BaseTool。
        """

        return LLMAgent(
            model_name=self.model_name,
            model_provider=self.model_provider,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=self.max_retries,
            trace_id=self.trace_id,

            _llm=self._llm,

            _tools=self._tools + list(tools),

            _struct_schema=self._struct_schema,
            _struct_tool_name=self._struct_tool_name,
            _output_instruction=self._output_instruction,
        )

    # ============================================================
    # Structured Output
    # ============================================================

    def with_struct(
        self,
        schema: Type[T],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> "LLMAgent":
        """
        使用 Tool Calling 实现结构化输出。

        注意：

        这里不会设置 tool_choice。

        这是为了兼容 DeepSeek thinking 模式。

        模型是否调用 output tool，
        由 System Prompt 引导。

        invoke() 后我们会严格检查：

            是否调用了 output tool
        """

        if not issubclass(schema, BaseModel):
            raise TypeError(
                "schema 必须是 Pydantic BaseModel 子类"
            )

        tool_name = name or schema.__name__

        output_instruction = (
            instruction
            or (
                f"最终结果必须通过 `{tool_name}` 工具返回。"
                f"不要直接使用普通文本输出最终结果。"
                f"调用 `{tool_name}` 时，参数必须严格符合工具 Schema。"
            )
        )

        return LLMAgent(
            model_name=self.model_name,
            model_provider=self.model_provider,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=self.max_retries,
            trace_id=self.trace_id,

            _llm=self._llm,

            _tools=self._tools,

            _struct_schema=schema,
            _struct_tool_name=tool_name,

            _output_instruction=output_instruction,
        )

    # ============================================================
    # 构造 Tool 列表
    # ============================================================

    def _build_tools(self) -> list[Any]:
        """
        返回最终发送给模型的 Tools。

        例如：

            with_tools([
                search_manual,
                query_timeseries,
            ])

            with_struct(DiagnosisResult)

        最终：

            [
                search_manual,
                query_timeseries,
                diagnosis_result_tool,
            ]
        """

        tools = list(self._tools)

        if self._struct_schema:

            output_tool = pydantic_to_tool(
                self._struct_schema,
                name=self._struct_tool_name,
            )

            tools.append(output_tool)

        return tools

    # ============================================================
    # 构造真正的模型
    # ============================================================

    def _build_bound_llm(self):
        """
        这里才真正调用 bind_tools()。

        注意：

        绝对不设置 tool_choice。

        因为：

            DeepSeek thinking
                +
            tool_choice

        可能发生冲突。

        所以 output tool 和普通 tool
        全部作为可用工具提供给模型。
        """

        tools = self._build_tools()

        if not tools:
            return self._llm

        return self._llm.bind_tools(
            tools
        )

    # ============================================================
    # System Prompt
    # ============================================================

    def _build_messages(
        self,
        messages: Any,
    ) -> list[Any]:

        if not isinstance(messages, list):
            messages = [messages]

        # 不修改调用者的 messages
        messages = list(messages)

        if not self._output_instruction:
            return messages

        # --------------------------------------------------------
        # 已有 SystemMessage
        # --------------------------------------------------------

        for index, message in enumerate(messages):

            if isinstance(message, SystemMessage):

                messages[index] = SystemMessage(
                    content=(
                        f"{message.content}\n\n"
                        f"{self._output_instruction}"
                    )
                )

                return messages

        # --------------------------------------------------------
        # 没有 SystemMessage
        # --------------------------------------------------------

        messages.insert(
            0,
            SystemMessage(
                content=self._output_instruction
            )
        )

        return messages

    # ============================================================
    # Structured Output 解析
    # ============================================================

    def _parse_struct(
        self,
        response: AIMessage,
    ) -> BaseModel:
        """
        从 AIMessage.tool_calls 中提取 output tool。

        注意：

        不关心普通工具调用。

        只寻找：

            self._struct_tool_name
        """

        if not self._struct_schema:
            raise RuntimeError(
                "当前没有绑定 Structured Schema"
            )

        tool_calls = response.tool_calls

        if not tool_calls:
            raise ValueError(
                "模型没有返回任何 tool_call，"
                f"期望调用 `{self._struct_tool_name}`"
            )

        # --------------------------------------------------------
        # 找 output tool
        # --------------------------------------------------------

        for tool_call in tool_calls:

            if (
                tool_call.get("name")
                != self._struct_tool_name
            ):
                continue

            args = tool_call.get("args")

            if args is None:
                raise ValueError(
                    f"output tool `{self._struct_tool_name}` "
                    "没有返回 args"
                )

            # ----------------------------------------------------
            # Pydantic 最终校验
            # ----------------------------------------------------

            return self._struct_schema.model_validate(
                args
            )

        # --------------------------------------------------------
        # 模型调用了普通工具，
        # 但没有调用 output tool
        # --------------------------------------------------------

        called_tools = [
            call.get("name")
            for call in tool_calls
        ]

        raise ValueError(
            f"模型没有调用最终输出工具 "
            f"`{self._struct_tool_name}`，"
            f"实际调用：{called_tools}"
        )

    # ============================================================
    # Invoke
    # ============================================================

    def invoke(
        self,
        messages: Any,
        *,
        trace_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:

        trace_id = trace_id or self.trace_id

        messages = self._build_messages(
            messages
        )

        llm = self._build_bound_llm()

        total_attempts = self.max_retries + 1

        last_error: Optional[Exception] = None

        # ========================================================
        # Retry
        # ========================================================

        for attempt in range(
            1,
            total_attempts + 1,
        ):

            start_time = time.perf_counter()

            logger.info(
                "LLM 请求开始 | "
                "trace_id=%s | "
                "model=%s | "
                "attempt=%s/%s",
                trace_id,
                self.model_name,
                attempt,
                total_attempts,
            )

            try:

                # ------------------------------------------------
                # LLM 调用
                # ------------------------------------------------

                response = llm.invoke(
                    messages,
                    **kwargs,
                )

                # ------------------------------------------------
                # Structured Output
                # ------------------------------------------------

                if self._struct_schema:

                    result = self._parse_struct(
                        response
                    )

                else:

                    result = response

                # ------------------------------------------------
                # 成功
                # ------------------------------------------------

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                logger.info(
                    "LLM 请求成功 | "
                    "trace_id=%s | "
                    "elapsed=%.3fs",
                    trace_id,
                    elapsed,
                )

                logger.debug(
                    "LLM 响应 | "
                    "trace_id=%s | "
                    "response=%s",
                    trace_id,
                    response,
                )

                # ------------------------------------------------
                # 持久化
                # ------------------------------------------------

                self._persist(
                    trace_id=trace_id,
                    messages=messages,
                    response=response,
                    result=result,
                )

                return result

            except Exception as e:

                last_error = e

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                logger.exception(
                    "LLM 请求失败 | "
                    "trace_id=%s | "
                    "attempt=%s/%s | "
                    "elapsed=%.3fs | "
                    "error=%s",
                    trace_id,
                    attempt,
                    total_attempts,
                    elapsed,
                    str(e),
                )

        # ========================================================
        # 所有重试失败
        # ========================================================

        raise last_error

    # ============================================================
    # Persistence
    # ============================================================

    def _persist(
        self,
        trace_id: str,
        messages: Any,
        response: Any,
        result: Any,
    ) -> None:
        """
        LLM 消息持久化扩展点。

        后续可以写入：

            trace_id
            messages
            AIMessage
            tool_calls
            result
            token_usage
            latency
        """

        pass

    # ============================================================
    # New Trace
    # ============================================================

    def new_trace(
        self,
        trace_id: Optional[str] = None,
    ) -> "LLMAgent":
        """
        创建新的 Trace。

        已经绑定的 Tools / Structured Schema 会保留。
        """

        return LLMAgent(
            model_name=self.model_name,
            model_provider=self.model_provider,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=self.max_retries,

            trace_id=(
                trace_id
                or uuid.uuid4().hex
            ),

            _llm=self._llm,
            _tools=self._tools,
            _struct_schema=self._struct_schema,
            _struct_tool_name=self._struct_tool_name,
            _output_instruction=self._output_instruction,
        )