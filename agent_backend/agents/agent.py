#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""A small, OpenAI-compatible agent loop.

The loop deliberately has no built-in skills or tools.  Callers can start with a
plain conversational agent and later add tools with :meth:`register_tool`.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from nce_service.llm_service import LLMService

logger = logging.getLogger(__name__)

_MAX_ROUNDS = 8
_DEFAULT_SYSTEM_PROMPT = Path(__file__).with_name("system_prompt.txt").read_text(
    encoding="utf-8"
)


class Agent:
    """Maintain conversation history and run an optional tool-calling loop."""

    def __init__(
            self,
            *,
            llm: LLMService | None = None,
            system_prompt: str | None = None,
            max_rounds: int = _MAX_ROUNDS,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self.llm = llm or LLMService.from_env()
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self.max_rounds = max_rounds
        self.messages: list[dict[str, Any]] = []
        self._tools: dict[str, tuple[dict[str, Any], Callable[..., Any]]] = {}

    @staticmethod
    def _parse_arguments(raw: str) -> dict[str, Any]:
        try:
            value = json.loads(raw or "{}")
            return value if isinstance(value, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _message_to_dict(message: Any) -> dict[str, Any]:
        if hasattr(message, "model_dump"):
            return message.model_dump(exclude_none=True)
        result = {"role": "assistant", "content": getattr(message, "content", None)}
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    def register_tool(
            self, schema: dict[str, Any], handler: Callable[..., Any]
    ) -> None:
        """Register an OpenAI function-tool schema and its sync/async handler."""
        try:
            name = schema["function"]["name"]
        except (KeyError, TypeError) as exc:
            raise ValueError("tool schema must contain function.name") from exc
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        self._tools[name] = (schema, handler)

    def reset(self) -> None:
        """Clear this agent's in-memory conversation."""
        self.messages.clear()

    async def chat(self, user_message: str) -> str:
        """Return the final text for one user turn."""
        chunks = []
        async for event in self.chat_stream(user_message):
            if event["type"] == "text":
                chunks.append(event["chunk"])
            elif event["type"] == "error":
                raise RuntimeError(event["message"])
        return "".join(chunks)

    async def chat_stream(self, user_message: str) -> AsyncGenerator[dict, None]:
        """Run one agent turn and yield text, tool-step, error, and done events."""
        if not user_message.strip():
            yield {"type": "error", "message": "用户消息不能为空"}
            return

        self.messages.append({"role": "user", "content": user_message})
        started = time.monotonic()

        for _ in range(self.max_rounds):
            try:
                response = await self._call_llm()
                message = response.choices[0].message
            except Exception as exc:
                logger.exception("LLM request failed")
                yield {"type": "error", "message": f"大模型调用失败: {exc}"}
                yield {"type": "done", "seconds": round(time.monotonic() - started, 1)}
                return

            assistant_message = self._message_to_dict(message)
            self.messages.append(assistant_message)
            tool_calls = getattr(message, "tool_calls", None) or []

            if not tool_calls:
                content = getattr(message, "content", None) or ""
                if content:
                    yield {"type": "text", "chunk": content}
                yield {"type": "done", "seconds": round(time.monotonic() - started, 1)}
                return

            for call in tool_calls:
                name = call.function.name
                arguments = self._parse_arguments(call.function.arguments)
                yield {"type": "step", "status": "running", "name": name, "args": arguments}
                result = await self._execute_tool(name, arguments)
                yield {"type": "step", "status": "done", "name": name, "result": result}
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

        yield {"type": "error", "message": f"工具调用超过 {self.max_rounds} 轮"}
        yield {"type": "done", "seconds": round(time.monotonic() - started, 1)}

    async def _call_llm(self):
        messages = [{"role": "system", "content": self.system_prompt}, *self.messages]
        kwargs: dict[str, Any] = {
            "model": self.llm.model,
            "messages": messages,
            "temperature": self.llm.temperature,
        }
        if self._tools:
            kwargs.update(
                tools=[item[0] for item in self._tools.values()],
                tool_choice="auto",
                parallel_tool_calls=False,
            )
        return await self.llm.client.chat.completions.create(**kwargs)

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        registered = self._tools.get(name)
        if registered is None:
            return json.dumps({"error": f"未注册工具: {name}"}, ensure_ascii=False)
        try:
            result = registered[1](**arguments)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
