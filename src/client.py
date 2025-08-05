# src/client.py

import asyncio
from fastmcp import Client
import yaml


async def main():
    url = "http://127.0.0.1:8000/mcp"
    # read config with LLM provider details and payload
    config_file = "config_test.yaml"
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    cfg["payload"]["config_file"] = config_file
    async with Client(url) as client:
        print(f"STATUS\tRun request one {cfg['payload']['data_folder']}")
        result = await client.call_tool("llm_pipeline", cfg["payload"])
        print(result)


asyncio.run(main())
