#!/usr/bin/env python3
"""Interactive smoke test for the agent.

Configure ``LLM_BASE_URL``, ``LLM_MODEL_NAME`` and optionally ``LLM_API_KEY``
in the environment or a .env file, then run: ``python agent_test.py``.
"""

import asyncio

from dotenv import load_dotenv

from agent_backend.agents.agent import Agent


async def main() -> None:
    load_dotenv()
    agent = Agent()
    print("Agent 已启动。输入 /reset 清空上下文，输入 /exit 退出。")

    while True:
        try:
            user_message = await asyncio.to_thread(input, "\n你: ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return

        command = user_message.strip().lower()
        if command in {"/exit", "/quit"}:
            print("再见！")
            return
        if command == "/reset":
            agent.reset()
            print("Agent: 上下文已清空。")
            continue
        if not command:
            continue

        print("Agent: ", end="", flush=True)
        async for event in agent.chat_stream(user_message):
            if event["type"] == "text":
                print(event["chunk"], end="", flush=True)
            elif event["type"] == "step":
                print(f"\n[{event['name']}: {event['status']}]", flush=True)
            elif event["type"] == "error":
                print(f"\n错误: {event['message']}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())