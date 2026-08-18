#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

"""
LLM 服务层，封装 OpenAI-compatible chat completions API（openai SDK）。

  stream_and_collect()  流式调用，实时打印，返回正式回答内容字符串
  complete()            流式调用，不打印，返回正式回答内容字符串
  complete_json()       complete() + JSON 解析

enable_thinking=True 时，服务端返回的 delta.reasoning_content 字段为思考过程，
delta.content 字段为正式回答；enable_thinking=False 时，仅有 delta.content。
两种情况下，返回值均为正式回答内容（reasoning_content 不计入返回值）。
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field

from openai import AsyncOpenAI


@dataclass
class LLMConfig:
    """
    单次 LLM 调用的运行时参数，可覆盖 LLMService 的全局默认值。

    None 表示"不覆盖，使用 LLMService 的默认值"。
    """

    model: str = ""
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout: int = 0
    max_retry: int = 2
    extra_payload: dict = field(default_factory=dict)

logger = logging.getLogger(__name__)
stream_logger = logging.getLogger("llm.stream")

# 单条 prompt 消息打日志时的截断长度（避免每次 LLM 调用都把整段 system prompt
# 写进日志造成膨胀）。设为 0 可关闭截断打印全文，调试时用。
_PROMPT_LOG_LIMIT = int(os.environ.get("PROMPT_LOG_LIMIT", "800"))

# 绕过代理直连 LLM 服务，避免内网地址被代理拦截
os.environ.setdefault("NO_PROXY", "oneapi.rnd.huawei.com")
os.environ.setdefault("no_proxy", "oneapi.rnd.huawei.com")


class LLMService:
    def __init__(
            self,
            base_url: str,
            model: str = "",
            api_key: str = "",
            temperature: float = 0.1,
            top_p: float = 1.0,
            timeout: int = 120,
            enable_thinking: bool = False,
            max_tokens: int = 4096,
    ):
        self._client = AsyncOpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url,
            timeout=timeout,
        )
        self.base_url = base_url
        self.default_model = model
        self._temperature = temperature
        self._top_p = top_p
        self._timeout = timeout
        self.enable_thinking = enable_thinking
        self._max_tokens = max_tokens

    @property
    def client(self):
        return self._client

    @property
    def model(self):
        return self.default_model

    @property
    def temperature(self):
        return self._temperature

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """
        从 LLM 输出中提取 JSON，兼容三种格式：
        1. ```json … ``` 代码块
        2. 裸 JSON 对象
        3. 文本中嵌套的 {...} 块
        """
        s = raw.strip()
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        first, last = s.find("{"), s.rfind("}")
        if first != -1 and last > first:
            try:
                return json.loads(s[first: last + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"无法从 LLM 输出中提取 JSON: {s[:200]}")

    @classmethod
    def from_env(cls) -> "LLMService":
        """从环境变量构造实例（需在调用前 load_dotenv）。"""
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
            model=os.getenv("LLM_MODEL_NAME", ""),
            api_key=os.getenv("LLM_API_KEY", ""),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
            top_p=float(os.getenv("LLM_TOP_P", 1.0)),
            timeout=int(os.getenv("LLM_TIMEOUT", 120)),
            enable_thinking=os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true",
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", 4096)),
        )

    async def stream_and_collect(
            self, messages: list[dict], config: LLMConfig | None = None
    ) -> str:
        """流式调用，实时打印到终端，返回正式回答内容字符串（不含思考内容）。"""
        return await self._stream(messages, config, print_stream=True)

    async def complete(
            self, messages: list[dict], config: LLMConfig | None = None
    ) -> str:
        """流式调用，不打印，返回正式回答内容字符串（不含思考内容）。"""
        return await self._stream(messages, config, print_stream=False)

    async def complete_json(
            self, messages: list[dict], config: LLMConfig | None = None
    ) -> dict:
        """complete() + 自动解析 JSON。"""
        raw = await self.complete(messages, config)
        return self._parse_json(raw)

    async def _stream(
            self,
            messages: list[dict],
            config: LLMConfig | None = None,
            print_stream: bool = False,
    ) -> str:
        cfg = config or LLMConfig()
        model, temperature, top_p, max_tokens = _resolve_config(
            cfg, self.default_model, self._temperature, self._top_p, self._max_tokens
        )

        _log_invoke(model, temperature, max_tokens, len(messages))
        _log_prompt(messages)

        stream = await self._create_stream(model, messages, temperature, top_p, max_tokens)
        answer_content, reasoning_content, finish_reason, usage = await _collect_stream(
            stream, print_stream
        )

        _log_output(answer_content, reasoning_content, finish_reason, usage)
        return _handle_empty_answer(answer_content, reasoning_content, finish_reason)

    async def _create_stream(
            self, model: str, messages: list[dict], temperature: float, top_p: float, max_tokens: int
    ):
        return await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
        )


def _resolve_config(
        cfg: LLMConfig, default_model: str, default_temperature: float,
        default_top_p: float, default_max_tokens: int
) -> tuple:
    model = cfg.model or default_model
    temperature = cfg.temperature if cfg.temperature is not None else default_temperature
    top_p = cfg.top_p if cfg.top_p is not None else default_top_p
    max_tokens = cfg.max_tokens if cfg.max_tokens is not None else default_max_tokens
    return model, temperature, top_p, max_tokens


def _log_invoke(model: str, temperature: float, max_tokens: int, msg_count: int) -> None:
    logger.info(
        "[LLM] 调用 model=%s temperature=%.2f max_tokens=%d messages=%d条",
        model, temperature, max_tokens, msg_count,
    )


def _log_prompt(messages: list[dict]) -> None:
    for m in messages:
        content = m.get("content")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        if _PROMPT_LOG_LIMIT and text and len(text) > _PROMPT_LOG_LIMIT:
            text = f"{text[:_PROMPT_LOG_LIMIT]}…（截断，共{len(text)}字，PROMPT_LOG_LIMIT=0 看全文）"
        logger.info("[LLM Prompt][%s]\n%s", m["role"], text)


async def _collect_stream(stream, print_stream: bool) -> tuple:
    reasoning_content = ""
    answer_content = ""
    is_answering = False
    finish_reason = None
    usage = None

    async for chunk in stream:
        usage = getattr(chunk, "usage", None) or usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        finish_reason = getattr(choice, "finish_reason", None) or finish_reason
        delta = choice.delta

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_content += reasoning
            if print_stream and not is_answering:
                stream_logger.info(reasoning)

        content = getattr(delta, "content", None)
        if content:
            if not is_answering:
                is_answering = True
            answer_content += content
            if print_stream:
                stream_logger.info(content)

    return answer_content, reasoning_content, finish_reason, usage


def _log_output(
        answer_content: str, reasoning_content: str, finish_reason: str, usage
) -> None:
    logger.info(
        "[LLM Output] (%d字，reasoning=%d字，finish_reason=%s，usage=%s):\n%s",
        len(answer_content), len(reasoning_content), finish_reason, usage, answer_content,
    )
    if reasoning_content:
        logger.info("[LLM reasoning_content 全文]\n%s", reasoning_content)


def _handle_empty_answer(
        answer_content: str, reasoning_content: str, finish_reason: str
) -> str:
    if not answer_content and reasoning_content:
        if finish_reason == "length":
            logger.warning(
                "[LLM] content 为空、reasoning_content 有 %d 字，finish_reason=length——"
                "输出被截断，可调大 LLM_MAX_TOKENS",
                len(reasoning_content),
            )
        else:
            logger.warning(
                "[LLM] content 为空但 reasoning_content 有 %d 字（finish_reason=%s，非截断），"
                "服务端可能未遵守 LLM_ENABLE_THINKING=false——回退使用 reasoning_content 作为回答",
                len(reasoning_content), finish_reason,
            )
            return reasoning_content
    return answer_content
