from datetime import datetime
from jinja2 import Template
import json
import logging
from numpy import nan
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path
import re
from typing import List
import yaml

from src.sysops import network
from src.sysops.filesystem import *
from src.sysops.names import *
from src.sysops import network
from src.scoring.base import RiskScoreFactory

from src.core import fastmcp_app

logger = logging.getLogger(__name__)

BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
}


def RESULT(msg):
    print(f"\033[92mRESULT\033[0m {msg}")


def WARN(msg):
    print(f"\033[91mWARNING\033[0m {msg}")


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
        try:
            return json.loads(last_json)
        except json.JSONDecodeError as e:
            logger.error(
                f"""
                    Could not decode JSON: {e}\nRaw block: {last_json}
                """
            )
            return {}


class Pipeline:

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._cfg["params"] = self._cfg.get("params") or {}
        kwargs = self._cfg["params"].pop("chat_template_kwargs", None)
        if kwargs:
            self._cfg["params"]["extra_body"] = {"chat_template_kwargs": kwargs}

        # Setup network proxy if configured
        self.original_proxies = network.set_proxy(cfg)

        # Setup outputs directories
        cfg = setup_directories(cfg)

        # self.outputs_base_dir = Path(cfg['outputs_dir'])
        self.log_base_dir = Path(cfg["log_dir"])
        self.stage1_dir = Path(cfg["stage1_dir"])
        self.stage2_dir = Path(cfg["stage2_dir"])

        # Collect keys for API calls
        keys = network.collect_api_keys(cfg)
        # init client
        match (cfg["provider"]):
            case "deepseek":
                self.client = OpenAI(
                    api_key=cfg["api_key"], base_url=BASE_URLS["deepseek"]
                )
            case "openai":
                self.client = OpenAI(
                    api_key=keys["api_key"],
                    organization=keys["org_key"],
                    project=keys["project_id"],
                )
            case "perplexity":
                self.client = OpenAI(
                    api_key=self.keys["api_key"], organization=keys["organization"]
                )
            case "qwen":
                self.client = OpenAI(
                    api_key=keys["api_key"], base_url=BASE_URLS["qwen"]
                )
            case _:
                raise ValueError(f"Unknown API: {cfg['provider']}")
        self.extractor = Extractor()

    def _collect_results(self, results_file1: Path, items: List):
        if results_file1.exists():
            results = pd.read_csv(results_file1, index_col="index")
        else:
            results = pd.DataFrame(columns=items)
        results = results.reindex(columns=results.columns.union(items))
        return results

    def call_llm(
        self, score_str: str, text: str, text_id: str, ts: str = None
    ) -> pd.Series:
        """
        Creates one prompt per item and returns after
        all items have been queried.
        """
        assert len(text) > 0, "Text must not be empty"
        # load template dictionary
        template_dict = load_prompt_template(score_str)
        system_prompt = load_system_prompt()
        items = [k for k in template_dict.keys() if k != "intro"]

        # setup log subdir for this case
        log_dir = self.log_base_dir / text_id
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger.info(f"Logs will be written to {log_dir}")

        # collect already inferred items
        results_file1 = self.stage1_dir / Path(f"{score_str}_llm.csv")
        results = self._collect_results(results_file1, items)

        # infer remaining
        if text_id not in results.index:
            results.loc[text_id] = [nan] * len(results.columns)

        results_row = results.loc[text_id]
        missing_items = results_row[results_row.isna()].index.tolist()
        jinja_template = Template(template_dict["intro"])
        prompt_intro = jinja_template.render(letter=text)

        for item in missing_items:
            logger.info(f"Processing {text_id}:\t{item}")
            prompt = " ".join([prompt_intro, template_dict[item]])

            # Call LLM processing routine
            response = self.call_per_item(prompt, system_prompt)
            # Log timestamped response
            save_log(item, log_dir, response, ts=ts)

            # Extract item value from response
            value_dict = self.extractor(response)
            if item in value_dict:
                results[item] = results[item].astype("object")
                results.at[text_id, item] = value_dict[item]
            else:
                logger.warning(
                    f"""
                        [call_llm] Unable to extract {item} from response:
                        {response}
                    """
                )

            # Save updated results
            results.reset_index().to_csv(results_file1, index=False)
            logger.info(
                f"""
                    [call_llm] Updated results for {text_id}:
                    {item} = {value_dict.get(item, 'not found')}
                """
            )

        if len(missing_items) > 1:
            logger.info(
                f"""
                    Inferred all items for {text_id}:
                    written to\t{results_file1}
                """
            )
        else:
            logger.info(
                f"""
                    Nothing to query, calculate final score
                """
            )
        return results.loc[text_id]

    def call_per_item(self, prompt, system_prompt) -> str:
        model = self._cfg["model"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        params = self._cfg["params"]
        completion = self.client.chat.completions.create(
            model=model, messages=messages, **params
        )
        is_stream = params.get("stream", False)
        if is_stream:
            completion_agg = []
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    completion_agg.append(chunk.choices[0].delta.content)
            return "".join(completion_agg)
        return completion.choices[0].message.content

    def call_calc(self, score_str: str, results_row: pd.DataFrame, text_id: str):

        # Dynamically call scoring function
        score = RiskScoreFactory.create(score_str.lower())
        score_items = score.calculate(results_row)

        # Load existing results file or initialize as empty
        calc_file = Path(f"{score_str}_calc.csv")
        self.results_file2 = self.stage2_dir / calc_file
        if self.results_file2.exists():
            scores_df = pd.read_csv(self.results_file2, index_col="index")
        else:
            scores_df = pd.DataFrame(columns=score_items.index)

        # Ensure text id is unique
        if text_id in scores_df.index:
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            text_id_new = f"{text_id}_{ts}"
            logger.warning(
                f"""
                    Score already computed for {text_id}, \
                    will update case under ID {text_id_new}
                """
            )
            text_id = text_id_new

        # Write processed items with final score back
        assert (score_items.index == scores_df.columns).all()
        scores_df.loc[text_id] = score_items
        scores_df.reset_index().to_csv(self.results_file2, index=False)
        logger.info(
            f"""
                Calculated scores written to\t{self.results_file2}
            """
        )
        return scores_df.loc[text_id]["score"]

    def tear_down(self):
        """
        Clear network proxy settings after processing.
        """
        network.clear_proxy(self.original_proxies)
        logger.info("Network proxies cleared.")


@fastmcp_app.tool()
def llm_pipeline(data_folder: str, risk_score: str, config_file: str) -> dict:
    """
    Single entry point for the LLM pipeline. For each case file in data_folder
    risk score items are extracted and the final score calculated as defined
    by physician rules and formula according to the original publication.
    Results of the first stage are collected in <score>_llm.csv and final
    calculated scores in <score>_calc.csv.

    accepted payload:
    {
        'data_folder': str, # path to folder with patient data
        'risk_score': str,  # one of HAS-BLED, CHA2DS2-VASc, or EuroSCORE II
        'config_file': str  # path to config.yaml with api key and model name
    }
    """
    # Get string representation of risk score
    score_str = get_score_str(risk_score)

    # Read configuration file
    assert os.path.isfile(config_file)
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    # Read patient data stored as txt file in data folder
    texts = read_text_files(data_folder)

    # Init pipeline
    pipeline = Pipeline(cfg)
    text_ids = sorted(texts.keys())
    for text_id in text_ids:
        text = texts[text_id]
        logger.info(f"Processing {text_id}")

        # 1st Step: LLM inference for raw items
        results_row = pipeline.call_llm(score_str, text, text_id)

        # 2nd Step: Apply physician rules and formula
        score = pipeline.call_calc(score_str, results_row, text_id)

        RESULT(f"{risk_score} of {text_id}:\t{score}")
