#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
class AgentWithSkills:
    """
agent
    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.registry = SkillRegistry(_SKILLS_DIR)
        self._loaded: set[str] = set()
        self.memory = AgentWithSkillsMemory()
        self._system_prompt = self._build_system_prompt()

    @staticmethod
    def _parse_tool_arguments(args_str: str) -> dict:
        """解析 tool call 参数；无效 JSON 按空参数处理。"""
        try:
            return json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            return {}

    # ── Public ────────────────────────────────────────────────────
    async def chat_stream(self, user_message: str) -> AsyncGenerator[dict, None]:
        """处理一轮用户输入，以事件流形式 yield 结果。"""
        self.memory.add_message({"role": "user", "content": user_message})
        t0 = time.time()

        for _ in range(_MAX_ROUNDS):
            response = await self._call_llm()
            choice = response.choices[0]
            msg = choice.message
            # 归一化 tool call：优先用服务端解析好的原生 tool_calls；
            # 若服务端 parser 失配把 tool call 当文本吐出来，则从 content 兜底解析。
            calls = self._normalize_tool_calls(choice, msg)

            if calls:

                async for event in self._stream_tool_calls(calls):
                    yield event
                continue

            if msg.content:
                yield {"type": "text", "chunk": msg.content}
            yield {"type": "done", "seconds": round(time.time() - t0, 1)}
            return

        yield {"type": "error", "message": "工具调用次数超限，请重试"}
        yield {"type": "done", "seconds": round(time.time() - t0, 1)}

    def reset(self) -> None:
        self.memory.reset()
        self._loaded.clear()
        _session_path(self.session_id).unlink(missing_ok=True)

    # ── Internal ──────────────────────────────────────────────────
    async def _stream_tool_calls(
            self, calls: list[tuple[str, str, str]]
    ) -> AsyncGenerator[dict, None]:
        """执行一轮 tool calls，并按执行顺序输出状态事件。"""
        for call_id, name, args_str in calls:
            args = self._parse_tool_arguments(args_str)
            logger.info("[Agent] tool=%s args=%s", name, args_str[:200])
            yield {
                "type": "step", "name": name, "status": "running",
                "call_id": call_id, "args": args,
            }

            result_dict, llm_str = await self._execute_tool(name, args)

            # bash 执行后推送检测到的状态变化事件
            for event in result_dict.get("_events", []):
                yield event

            yield {
                "type": "step", "name": name, "status": "done",
                "call_id": call_id,
                "result": _result_display(name, result_dict, llm_str),
                "detail": llm_str,
            }
            self.memory.add_message(
                {"role": "tool", "tool_call_id": call_id, "content": llm_str}
            )

    def _build_system_prompt(self) -> str:
        lines = []
        for m in self.registry.list_all():
            cat = f"[{m['category']}] " if m.get("category") else ""
            lines.append(f"- {cat}{m['name']}: {m.get('description', '')}")
        skill_block = _SKILL_SYSTEM_TEMPLATE.format(skill_entries="\n".join(lines))
        return f"{_SYSTEM_PROMPT}\n\n{skill_block}"

    async def _call_llm(self):
        llm = LLMService.from_env()
        messages = self.memory.build_messages(self._system_prompt)
        logger.info("[Agent._call_llm] messages=%d", len(messages))
        return await llm.client.chat.completions.create(
            model=llm.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=llm.temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    def _normalize_tool_calls(self, choice, msg) -> list[tuple[str, str, str]]:
        """归一化本轮的 tool call，并把 assistant 消息写入 memory。

        返回 [(call_id, name, arguments_str), ...]；无 tool call 时返回 []。
        优先用服务端原生 tool_calls；服务端 parser 失配时从 content 文本兜底解析。
        """
        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            self.memory.add_message(msg.model_dump(exclude_none=True))
            return [(tc.id, tc.function.name, tc.function.arguments) for tc in msg.tool_calls]

        text_calls = _parse_text_tool_calls(msg.content or "")
        if not text_calls:
            self.memory.add_message(msg.model_dump(exclude_none=True))
            return []

        # 文本兜底：重写 assistant 消息为规范 tool_calls 形态并清掉泄漏的原始文本，
        # 否则后续 role:tool 引用的 tool_call_id 在历史中找不到，部分模板会报错。
        synth_tcs, calls = [], []
        for i, c in enumerate(text_calls):
            cid = f"call_text_{i}_{uuid.uuid4().hex[:8]}"
            args_str = json.dumps(c["arguments"], ensure_ascii=False)
            synth_tcs.append({
                "id": cid, "type": "function",
                "function": {"name": c["name"], "arguments": args_str},
            })
            calls.append((cid, c["name"], args_str))
        self.memory.add_message({"role": "assistant", "content": None, "tool_calls": synth_tcs})
        logger.warning("[Agent] 服务端未返回原生 tool_calls，已从文本兜底解析 %d 个: %s",
                       len(calls), [c[1] for c in calls])
        return calls
