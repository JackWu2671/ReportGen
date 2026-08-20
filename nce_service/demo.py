#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import asyncio
import logging

from dotenv import load_dotenv

from llm_service import LLMService

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


async def main():
    load_dotenv()

    messages = [{"role": "user", "content": "你好"}]
    # 在事件循环结束前关闭底层 HTTP 连接池，避免异步生成器在解释器退出阶段才被回收。
    async with LLMService.from_env() as llm:
        await llm.stream_and_collect(
            messages,
            on_content=lambda content: print(content, end="", flush=True),
        )


if __name__ == "__main__":
    asyncio.run(main())
