#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""A small, OpenAI-compatible agent loop.

The loop deliberately has no built-in skills or tools.  Callers can start with a
plain conversational agent and later add tools with :meth:`register_tool`.
"""

from __future__ import annotations

import inspect
import os
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from nce_service.llm_service import LLMService
from skill_registry import SkillRegistry

logger = logging.getLogger(__name__)

_MAX_ROUNDS = 8
_DEFAULT_SYSTEM_PROMPT = Path(__file__).with_name("system_prompt.txt").read_text(
    encoding="utf-8"
)
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_AGENT_DIR)
_SKILLS_DIR = Path(_BACKEND_DIR) / "skills"
_SYSTEM_PROMPT = (Path(_AGENT_DIR) / "system_prompt.txt").read_text(encoding="utf-8")
_SKILL_SYSTEM_TEMPLATE = """\
<skill_system>
遇到复杂任务先用 read_skill(<skill_name>) 阅读工作流指导，再用 bash 执行对应脚本。
只在需要时读取，不要预先读取所有技能。

<available_skills>
{skill_entries}
</available_skills>
</skill_system>"""


class Agent:
    """Maintain conversation history and run an optional tool-calling loop."""

    def __init__(
            self,
            *,
            session_id: str = "",
            llm: LLMService | None = None,
            system_prompt: str | None = None,
            max_rounds: int = _MAX_ROUNDS,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self.skills = SkillRegistry(_SKILLS_DIR)
        self.llm = llm or LLMService.from_env()
        self.system_prompt = self._build_system_prompt()

        self.session_id = session_id or str(uuid.uuid4())
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
    def _merge_tool_call(
            tool_calls_by_index: dict[int, dict[str, Any]], call: Any
    ) -> None:
        merged = tool_calls_by_index.setdefault(
            call.index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if call.id:
            merged["id"] = call.id
        function = getattr(call, "function", None)
        if function is not None:
            merged["function"]["name"] += getattr(function, "name", None) or ""
            merged["function"]["arguments"] += getattr(function, "arguments", None) or ""

    @staticmethod
    def _done_event(started: float) -> dict[str, Any]:
        return {"type": "done", "seconds": round(time.monotonic() - started, 1)}

    def _build_system_prompt(self) -> str:
        lines = []
        for m in self.skills.list_all():
            cat = f"[{m['category']}] " if m.get("category") else ""
            lines.append(f"- {cat}{m['name']}: {m.get('description', '')}")
        skill_block = _SKILL_SYSTEM_TEMPLATE.format(skill_entries="\n".join(lines))
        return f"{_SYSTEM_PROMPT}\n\n{skill_block}"

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
            assistant_message = None
            tool_calls = []
            async for response_event in self._request_events():
                if response_event["type"] == "text":
                    yield response_event
                    continue
                if response_event["type"] == "error":
                    yield response_event
                    yield self._done_event(started)
                    return
                assistant_message = response_event["message"]
                tool_calls = response_event["tool_calls"]

            if assistant_message is None:
                yield {"type": "error", "message": "大模型响应不完整"}
                yield self._done_event(started)
                return

            self.messages.append(assistant_message)

            if not tool_calls:
                yield self._done_event(started)
                return

            async for tool_event in self._run_tools(tool_calls):
                yield tool_event

        yield {"type": "error", "message": f"工具调用超过 {self.max_rounds} 轮"}
        yield self._done_event(started)

    async def _call_llm(self):
        messages = [{"role": "system", "content": self.system_prompt}, *self.messages]
        kwargs: dict[str, Any] = {
            "model": self.llm.model,
            "messages": messages,
            "temperature": self.llm.temperature,
            "top_p": self.llm.top_p,
            "max_tokens": self.llm.max_tokens,
            # Agent 直接使用底层 client，必须与 LLMService._create_stream 保持一致；
            # 否则 Qwen 服务端可能采用默认的 thinking 模式，在正式 content 前先生成
            # 一段不可见 reasoning，表现为首字等待数秒。
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": self.llm.enable_thinking}
            },
        }
        if self._tools:
            kwargs.update(
                tools=[item[0] for item in self._tools.values()],
                tool_choice="auto",
                parallel_tool_calls=False,
            )
        return await self.llm.client.chat.completions.create(**kwargs, stream=True)

    async def _iter_response(self, stream) -> AsyncGenerator[dict[str, Any], None]:
        """逐块转发文本，并将工具调用增量合并成一条 assistant 消息。"""
        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                content = getattr(delta, "content", None)
                if content:
                    content_parts.append(content)
                    yield {"type": "text", "chunk": content}

                for call in getattr(delta, "tool_calls", None) or []:
                    self._merge_tool_call(tool_calls_by_index, call)
        finally:
            await stream.close()

        tool_calls = [tool_calls_by_index[index] for index in sorted(tool_calls_by_index)]
        content = "".join(content_parts)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            if not content:
                message["content"] = None
            message["tool_calls"] = tool_calls
        yield {"type": "response", "message": message, "tool_calls": tool_calls}

    async def _request_events(self) -> AsyncGenerator[dict[str, Any], None]:
        try:
            stream = await self._call_llm()
            async for event in self._iter_response(stream):
                yield event
        except Exception as exc:
            logger.exception("LLM request failed")
            yield {"type": "error", "message": f"大模型调用失败: {exc}"}

    async def _run_tools(
            self, tool_calls: list[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        for call in tool_calls:
            name = call["function"]["name"]
            arguments = self._parse_arguments(call["function"]["arguments"])
            yield {"type": "step", "status": "running", "name": name, "args": arguments}
            result = await self._execute_tool(name, arguments)
            yield {"type": "step", "status": "done", "name": name, "result": result}
            self.messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )

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
