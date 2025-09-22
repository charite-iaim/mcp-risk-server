# /tests/pipeline/test_integration.py

import os
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
import pandas.testing as pdt
from pathlib import Path
import pytest
import tempfile
from unittest.mock import patch
import yaml

from src.core import fastmcp_app
from src.pipeline.provider_tools import llm_pipeline, Pipeline
from src.sysops.filesystem import _get_repo_root, setup_directories

from fastmcp import FastMCP, Client


@pytest.fixture
def mcp_server():
    return fastmcp_app


@pytest.fixture
def default_cfg():
    with tempfile.TemporaryDirectory() as tempdir:
        return {
            "run_name": "test_run",
            "risk_score": "hasbled",
            "provider": "deepseek",
            "network": {"http_proxy": None, "https_proxy": None},
            "model": "mock-model",
            "outputs_dir": str(Path(tempdir) / Path("outputs")),
            "api_key": "mock-api-key",
            "org_key": None,
            "project_id": None,
            "recalculate": False,
        }


@pytest.fixture
def hasbled_llm_P4ab463aa():
    return {
        "sys_bp": ('{\n"sys_bp": 140\n}', 140),
        "renal_disease": ('{\n"renal_disease": 0\n}', 0),
        "creatinine": ('{\n"creatinine": "missing"\n}', "missing"),
        "unit_mg_per_dL": ('{\n"unit_mg_per_dL": 0\n}', 0),
        "liver_disease": ('{\n"liver_disease": 0\n}', 0),
        "ast": ('{\n"ast": 100\n}', 100),
        "alt": ('{\n"alt": 90\n}', 90),
        "alp": ('{\n"alp": 300\n}', 300),
        "bilirubin": ('{\n"bilirubin": 2.5\n}', 2.5),
        "stroke_history": ('{\n"stroke_history": 0\n}', 0),
        "bleeding_history": ('{\n"bleeding_history": 0\n}', 0),
        "hemoglobin": ('{\n"hemoglobin": 15\n}', 15),
        "is_female": ('{\n"is_female": 0\n}', 0),
        "labile_inr": ('{\n"labile_inr": 0\n}', 0),
        "date_of_birth": ('{\n"date_of_birth": "01.02.1960"\n}', "01.02.1960"),
        "date_of_discharge": ('{\n"date_of_discharge": "02.02.2020"\n}', "02.02.2020"),
        "med_bleed": ('{\n"med_bleed": 0\n}', 0),
        "alcohol": ('{\n"alcohol": 0\n}', 0),
    }  # HAS-BLED: 0


@pytest.fixture
def hasbled_calc_P4ab463aa():
    return Series(
        {
            "index": "P4ab463aa",
            "H": 0,
            "A1": 0,
            "A2": 0,
            "S": 0,
            "B": 0,
            "L": 0,
            "E": 0,
            "D1": 0,
            "D2": 0,
            "score": 0,
        }
    )


@pytest.fixture
def hasbled_llm_Pb4cf5ebc():
    return {
        "sys_bp": ('{\n"sys_bp": 170\n}', 170),
        "renal_disease": ('{\n"renal_disease": 1\n}', 1),
        "creatinine": ('{\n"creatinine": "missing"\n}', "missing"),
        "unit_mg_per_dL": ('{\n"unit_mg_per_dL": 1\n}', 1),
        "liver_disease": ('{\n"liver_disease": 1\n}', 1),
        "ast": ('{\n"ast": "missing"\n}', "missing"),
        "alt": ('{\n"alt": 223\n}', 223),
        "alp": ('{\n"alp": 415\n}', 415),
        "bilirubin": ('{\n"bilirubin": 4.6\n}', 4.6),
        "stroke_history": ('{\n"stroke_history": 1\n}', 1),
        "bleeding_history": ('{\n"bleeding_history": 1\n}', 1),
        "hemoglobin": ('{\n"hemoglobin": 11.4\n}', 11.4),
        "is_female": ('{\n"is_female": 0\n}', 0),
        "labile_inr": ('{\n"labile_inr": 1\n}', 1),
        "date_of_birth": ('{\n"date_of_birth": "22.06.1950"\n}', "22.06.1950"),
        "date_of_discharge": ('{\n"date_of_discharge": "01.07.2025"\n}', "01.07.2025"),
        "med_bleed": ('{\n"med_bleed": 1\n}', 1),
        "alcohol": ('{\n"alcohol": 1\n}', 1),
    }  # HAS-BLED: 9


@pytest.fixture
def hasbled_calc_Pb4cf5ebc():
    return Series(
        {
            "index": "Pb4cf5ebc",
            "H": 1,
            "A1": 1,
            "A2": 1,
            "S": 1,
            "B": 1,
            "L": 1,
            "E": 1,
            "D1": 1,
            "D2": 1,
            "score": 9,
        }
    )


def save_dict_as_yaml(d, file_path):
    with open(file_path, "w") as file:
        yaml.dump(d, file, sort_keys=False)


def test_checkpointing_llm_proceed_with_next_case1(
    default_cfg, hasbled_llm_P4ab463aa, hasbled_calc_P4ab463aa
):
    cfg = default_cfg.copy()
    # Test checkpointing with incompletely processed reports:
    # first report completed, second report not started yet
    text_id1 = "P4ab463aa"  # already queried and calculated
    text_id2 = "Pb4cf5ebc"  # completely unprocessed
    score_str = cfg["risk_score"]
    # prepare env: store cfg in tmp dir
    cfg = default_cfg.copy()
    data_folder = Path(_get_repo_root()) / "tests/data/reports_hasbled"
    assert os.path.exists(data_folder)
    risk_score = "HAS-BLED"
    with tempfile.TemporaryDirectory() as tempdir:
        cfg_file = Path(tempdir) / "cfg.yaml"
        save_dict_as_yaml(cfg, cfg_file)

    # setup stage output dirs
    cfg_setup = cfg.copy()
    cfg_setup = setup_directories(cfg_setup)

    # store query extracts of text report
    # data/reports_hasbled/P4ab463aa_0.txt
    df_llm = DataFrame({c: [v[1]] for c, v in hasbled_llm_P4ab463aa.items()})
    df_llm["index"] = text_id1
    results_file1 = Path(cfg_setup["stage1_dir"]) / Path(f"{score_str}_llm.csv")
    df_llm.to_csv(results_file1, index=False)

    # store calculated items of first report
    df_calc = DataFrame({c: [v] for c, v in hasbled_calc_P4ab463aa.items()})
    results_file2 = Path(cfg_setup["stage2_dir"]) / Path(f"{score_str}_calc.csv")
    df_calc.to_csv(results_file2, index=False)

    # test collected results as expected
    pipeline = Pipeline(cfg)
    results1 = pipeline._collect_results(
        results_file1, items=hasbled_llm_P4ab463aa.keys()
    )
    results1_tgt = DataFrame({k: [v[1]] for k, v in hasbled_llm_P4ab463aa.items()})
    results1_tgt["index"] = [text_id1]
    results1_tgt.set_index("index", inplace=True)

    pdt.assert_frame_equal(results1_tgt, results1, check_like=True)

    # analogously for calculated results
    results2 = pipeline._collect_results(
        results_file2, items=hasbled_calc_P4ab463aa.keys()
    )
    results2_tgt = DataFrame({k: [v] for k, v in hasbled_calc_P4ab463aa.items()})
    results2_tgt.set_index("index", inplace=True)

    pdt.assert_frame_equal(results2_tgt, results2, check_like=True)


@pytest.mark.asyncio
async def test_checkpointing_llm_proceed_with_next_case2(
    mcp_server,
    default_cfg,
    hasbled_llm_P4ab463aa,
    hasbled_calc_P4ab463aa,
    hasbled_llm_Pb4cf5ebc,
    hasbled_calc_Pb4cf5ebc,
):
    # assume preceeding test succeeds, ie, loading checkpointed
    # case P4ab463aa behaves as expected
    cfg = default_cfg.copy()
    # Test checkpointing with incompletely processed reports:
    # first report completed, second report not started yet
    text_id1 = "P4ab463aa"  # already queried and calculated
    text_id2 = "Pb4cf5ebc"  # completely unprocessed
    score_str = cfg["risk_score"]
    # prepare env: store cfg in tmp dir
    cfg = default_cfg.copy()
    data_folder = Path(_get_repo_root()) / "tests/data/reports_hasbled"
    risk_score = "HAS-BLED"
    with tempfile.TemporaryDirectory() as tempdir:
        cfg_file = Path(tempdir) / "cfg.yaml"
        save_dict_as_yaml(cfg, cfg_file)
        assert os.path.isfile(cfg_file)

        # setup stage output dirs
        cfg_setup = cfg.copy()
        cfg_setup = setup_directories(cfg_setup)

        # store query extracts of text report
        # data/reports_hasbled/P4ab463aa_0.txt
        df_llm = DataFrame({c: [v[1]] for c, v in hasbled_llm_P4ab463aa.items()})
        df_llm["index"] = text_id1
        results_file1 = Path(cfg_setup["stage1_dir"]) / Path(f"{score_str}_llm.csv")
        df_llm.to_csv(results_file1, index=False)

        # store calculated items of first report
        df_calc = DataFrame({c: [v] for c, v in hasbled_calc_P4ab463aa.items()})
        results_file2 = Path(cfg_setup["stage2_dir"]) / Path(f"{score_str}_calc.csv")
        df_calc.to_csv(results_file2, index=False)

        # test collected results as expected
        pipeline = Pipeline(cfg)

        # call_per_item is patched, list of mocked_responses iterated instead
        mocked_responses = [v[0] for v in hasbled_llm_Pb4cf5ebc.values()]
        with patch(
            "src.pipeline.provider_tools.Pipeline.call_per_item",
            side_effect=mocked_responses,
        ):
            async with Client(mcp_server) as client:
                _ = await client.call_tool(
                    "llm_pipeline",
                    {
                        "data_folder": data_folder,
                        "risk_score": risk_score,
                        "config_file": cfg_file,
                    },
                )

                # now test if checkpointed results_file1/2 have been augmented with case 2
                results1_post = pipeline._collect_results(
                    results_file1, items=hasbled_llm_P4ab463aa.keys()
                )

                results1_post_tgt = DataFrame(
                    {
                        k: [hasbled_llm_P4ab463aa[k][1], hasbled_llm_Pb4cf5ebc[k][1]]
                        for k in hasbled_llm_P4ab463aa.keys()
                    }
                )
                results1_post_tgt["index"] = [text_id1, text_id2]
                results1_post_tgt.set_index("index", inplace=True)

                dfs = [results1_post, results1_post_tgt]
                pd.set_option("future.no_silent_downcasting", True)
                for i, df in enumerate(dfs):
                    df = df.replace("missing", np.nan).infer_objects(copy=False)
                    df = df.apply(pd.to_numeric, errors="coerce")
                    df = df.astype("float64")
                    dfs[i] = df
                results1_post, results1_post_tgt = dfs

                pdt.assert_frame_equal(results1_post_tgt, results1_post)

                # Now compare stage 2 results, ie, calculated items and final score
                results2_post = pipeline._collect_results(
                    results_file2, items=hasbled_calc_P4ab463aa.keys()
                )
                results2_post_tgt = DataFrame(
                    [hasbled_calc_P4ab463aa, hasbled_calc_Pb4cf5ebc]
                )
                results2_post_tgt.set_index("index", inplace=True)

                # harmonize stage 2 dataframes
                results2_post = results2_post.astype("float64")
                results2_post_tgt = results2_post_tgt.astype("float64")

                pdt.assert_frame_equal(results2_post_tgt, results2_post)
