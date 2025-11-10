# src/client.py
import argparse
import asyncio
from fastmcp import Client
from pathlib import Path
import yaml


from src.sysops.filesystem import get_repo_root

DATA_DEFAULT = Path(get_repo_root()) / "data"
CFG_DEFAULT = Path(get_repo_root()) / "my_config.yaml"
URL_DEFAULT = "http://127.0.0.1:8000/mcp"


parser = argparse.ArgumentParser(
    prog="Client", description="Uses the MCP server for risk score calculation"
)

parser.add_argument(
    "--data",
    type=str,
    default=DATA_DEFAULT,
    help="Set data input directory",
)

parser.add_argument(
    "--cfg",
    type=str,
    default=CFG_DEFAULT,
    help="Give configuration file",
)

parser.add_argument(
    "--url",
    type=str,
    default=URL_DEFAULT,
    help="Give MCP server URL",
)


async def main():
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    async with Client(args.url) as client:
        print(f"STATUS\tRun {cfg['risk_score']} for data folder: {args.data}")
        result = await client.call_tool(
            "llm_pipeline", {"data_folder": args.data, "config_file": args.cfg}
        )


asyncio.run(main())
