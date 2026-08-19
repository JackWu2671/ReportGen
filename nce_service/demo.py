#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging

from dotenv import load_dotenv

from llm_service import LLMService

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


async def main():
    load_dotenv()

    messages = [{"role": "user", "content": "我希望你写一个300字的网络分析经验"}]
    # 在事件循环结束前关闭底层 HTTP 连接池，避免异步生成器在解释器退出阶段才被回收。
    async with LLMService.from_env() as llm:
        await llm.stream_and_collect(messages)


if __name__ == "__main__":
    asyncio.run(main())
