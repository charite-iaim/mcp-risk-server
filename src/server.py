# src/server.py

import argparse
import logging
import os

from src.core import fastmcp_app
from src.pipeline.provider_tools import llm_pipeline


parser = argparse.ArgumentParser(prog="Server", description="Runs the local MCP server")

parser.add_argument(
    "-l",
    "--log_level",
    type=str,
    default="INFO",
    help="Set log level to either: INFO [default], DEBUG, WARNING, or ERROR",
)


def configure_logging(level: str = "INFO"):
    level = level.upper()
    assert level in logging._nameToLevel
    logging.basicConfig(
        level=logging._nameToLevel[level],
        format="[%(levelname)s] %(name)s: %(message)s",
    )


if __name__ == "__main__":
    args = parser.parse_args()
    configure_logging(level=args.log_level)

    print("Starting FastMCP Server with HTTP transport layer ...")
    fastmcp_app.run(transport="http", host="127.0.0.1", port=8000)
    print("Shut down FastMCP Server.")
