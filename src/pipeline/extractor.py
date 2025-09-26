# src/pipeline/extractor.py
import json
import logging
import re

logger = logging.getLogger(__name__)

class Extractor:

    def __init__(self):
        self.json_rx = re.compile(r"\{.*?\}", re.DOTALL)

    def __call__(self, text_with_json: str) -> dict:
        # Search for all possible JSON blocks and get the last one
        matches = self.json_rx.findall(text_with_json)
        if not matches:
            logger.error(f"Could not extract JSON block: {text_with_json}")
            return {}
        last_json = matches[-1]
        # unescape raw symbols
        last_json = last_json.encode('utf-8').decode('unicode_escape')
        # remove single quotes
        if last_json.startswith("'") and last_json.endswith("'"):
            last_json = last_json[1:-1]
        try:
            return json.loads(last_json)
        except json.JSONDecodeError as e:
            logger.error(
                f"""
                    Could not decode JSON: {e}\nRaw block: {last_json}
                """
            )
            return {}
