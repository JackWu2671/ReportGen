#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging

from dotenv import load_dotenv

from llm_service import LLMService

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


async def main():
    load_dotenv()

    messages = [{"role": "user", "content": "你好，你是是谁？"}]
    # 在事件循环结束前关闭底层 HTTP 连接池，避免异步生成器在解释器退出阶段才被回收。
    async with LLMService.from_env() as llm:
        answer = await llm.stream_and_collect(messages)

    print(f"\n最终回答: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
