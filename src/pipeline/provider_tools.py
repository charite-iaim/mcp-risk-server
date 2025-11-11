# /src/pipeline/provider_tools.py

from datetime import datetime
from jinja2 import Template
import json
import logging
from numpy import nan
from openai import OpenAI
from perplexity import Perplexity
import os
import pandas as pd
from pathlib import Path
import re
from typing import List
import yaml


from src.core import fastmcp_app
from src.pipeline.extractor import Extractor
from src.sysops import network
from src.sysops.filesystem import *
from src.sysops.names import *
from src.sysops import network
from src.scoring.base import RiskScoreFactory
from src.scoring.cha2ds2vasc import *
from src.scoring.euroscoreii import *
from src.scoring.hasbled import *


logger = logging.getLogger(__name__)

BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "perplexity": "https://api.perplexity.ai",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
}


class Pipeline:

    def __init__(self, cfg: dict) -> None:
        # get string representation of risk score
        self.score_str = get_score_str(cfg["risk_score"])

        self._cfg = cfg
        self._cfg["params"] = self._cfg.get("params") or {}
        kwargs = self._cfg["params"].pop("chat_template_kwargs", None)
        if kwargs:
            self._cfg["params"]["extra_body"] = {"chat_template_kwargs": kwargs}

        # Setup network proxy if configured
        self.original_proxies = network.set_proxy(cfg)

        # Setup output directories
        cfg = setup_directories(cfg)

        # self.output_base_dir = Path(cfg['output_dir'])
        self.log_base_dir = Path(cfg["log_dir"])
        self.stage1_dir = Path(cfg["stage1_dir"])
        self.stage2_dir = Path(cfg["stage2_dir"])

        self.results_file1 = self.stage1_dir / Path(f"{self.score_str}_llm.csv")

        self.results_file2 = self.stage2_dir / Path(f"{self.score_str}_calc.csv")

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
                self.client = Perplexity(
                    api_key=keys["api_key"], base_url=BASE_URLS["perplexity"]
                )
            case "qwen":
                self.client = OpenAI(
                    api_key=keys["api_key"], base_url=BASE_URLS["qwen"]
                )
            case _:
                raise ValueError(f"Unknown API: {cfg['provider']}")
        self.extractor = Extractor()

    def _collect_results(self, results_file: Path, items: List):
        # return dataframe index on a column named "index"
        # in case datafile exists possibly missing columns will be added,
        # otherwise an empty frame is constructed
        items = [item for item in items if item != "index"]
        if results_file.exists():
            results = pd.read_csv(results_file, index_col="index")
        else:
            items = [item for item in items if item != "index"]
            results = pd.DataFrame(columns=items)
            results.index.name = "index"

        # augment with any missing columns from 'items' and fill with NaN
        for col in items:
            if col not in results.columns:
                results[col] = nan

        results = results.reindex(columns=items)
        return results

    def call_llm(self, text: str, text_id: str, ts: str = None) -> pd.Series:
        """
        Creates one prompt per item and returns after
        all items have been queried.
        """
        assert len(text) > 0, "Text must not be empty"
        # load template dictionary
        template_dict = load_prompt_template(self.score_str)
        system_prompt = load_system_prompt()
        items = [k for k in template_dict.keys() if k != "intro"]

        # setup log subdir for this case
        log_dir = self.log_base_dir / text_id
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger.info(f"Logs will be written to {log_dir}")

        # collect already inferred items
        results = self._collect_results(self.results_file1, items)

        # infer remaining
        if text_id not in results.index:
            results.loc[text_id, :] = [nan] * len(items)

        results_row = results.loc[text_id]
        # query behaviour: items already queried, but failed to be
        # extracted will be queried again.
        missing_items = results_row[results_row.isna()].index.tolist() + [
            item for item in items if not item in results_row.index
        ]
        jinja_template = Template(template_dict["intro"])
        prompt_intro = jinja_template.render(letter=text)

        for item in missing_items:
            logger.info(f"Processing {text_id}:\t{item}")
            prompt = " ".join([prompt_intro, template_dict[item]])

            # call LLM processing routine
            response = self.call_per_item(prompt, system_prompt)

            # Log timestamped response
            save_log(item, log_dir, response, ts=ts)

            # extract item value from response
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

            # save updated results
            results.reset_index().to_csv(self.results_file1, index=False)
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
                    written to\t{self.results_file1}
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

    def call_calc(self, results_row: pd.DataFrame, text_id: str) -> pd.Series:

        # dynamically call scoring function
        score = RiskScoreFactory.create(self.score_str.lower())
        score_items = score.calculate(results_row)

        # load existing results file or initialize as empty
        items = score_items.index
        scores_df = self._collect_results(self.results_file2, items)

        # ensure text id is unique
        if text_id in scores_df.index and not pd.isna(scores_df.at[text_id, "score"]):
            if not self._cfg.get("recalculate", False):
                score = scores_df.at[text_id, "score"]
                logger.debug(
                    f"""
                    Score already computed for {text_id}: {score}, \
                    will skip recalculation step as set in config.
                """
                )
                return scores_df.loc[text_id]
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            text_id_new = f"{text_id}_{ts}"
            logger.debug(
                f"""
                    Score already computed for {text_id}, will recalculate and update case under ID {text_id_new}
                """
            )
            text_id = text_id_new

        # write processed items with final score back
        assert (score_items.index == scores_df.columns).all()
        scores_df.loc[text_id] = score_items
        scores_df.reset_index().to_csv(self.results_file2, index=False)

        return scores_df.loc[text_id]

    def tear_down(self):
        """
        Clear network proxy settings after processing.
        """
        network.clear_proxy(self.original_proxies)
        logger.info("Network proxies cleared.")


def _llm_pipeline_inner(data_folder: str, config_file: str) -> dict:
    # read configuration file
    assert os.path.isfile(config_file)
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    print("read config in inner pipeline:\n", cfg)

    # read patient data stored as txt file in data folder
    texts = read_text_files(data_folder)

    # init pipeline
    pipeline = Pipeline(cfg)
    text_ids = sorted(texts.keys())
    print("text ids: ", text_ids)
    results = []
    for text_id in text_ids:
        text = texts[text_id]
        logger.info(f"Processing {text_id}")

        # 1st Step: LLM inference for raw items
        results_row = pipeline.call_llm(text, text_id)

        # 2nd Step: Apply physician rules and formula
        result = pipeline.call_calc(results_row, text_id)
        results.append(result)

        logger.info(f"{cfg['risk_score']} of {text_id}:\t{result['score']}")

    logger.info(
        f"""
            Calculated scores written to\t{pipeline.results_file2}
        """
    )
    df_as_dict = pd.DataFrame(results).to_dict(orient="records")
    return {"results": df_as_dict}


@fastmcp_app.tool()
def llm_pipeline(data_folder: str, config_file: str) -> dict:
    """
    Single entry point for the LLM pipeline. For each case file in data_folder
    risk score items are extracted and the final score calculated as defined
    by physician rules and formula according to the original publication.
    Results of the first stage are collected in <score>_llm.csv and final
    calculated scores in <score>_calc.csv.

    accepted payload:
    {
        'data_folder': str, # path to folder with patient data
        'config_file': str  # path to config.yaml with api key and model name
    }
    """
    return _llm_pipeline_inner(data_folder, config_file)
