#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging

from dotenv import load_dotenv

from llm_service import LLMService

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


async def main():
    load_dotenv()

    llm = LLMService.from_env()
    messages = [{"role": "user", "content": "你好，你是是谁？"}]

    answer = await llm.stream_and_collect(messages)
    print(f"\n最终回答: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
