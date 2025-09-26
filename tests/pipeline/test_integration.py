# /tests/pipeline/test_integration.py

from fastmcp import Client
import os
import numpy as np
from pandas import DataFrame
import pandas as pd
import pandas.testing as pdt
from pathlib import Path
import pytest
import tempfile
from unittest.mock import patch
import yaml

from src.core import fastmcp_app
from src.pipeline.provider_tools import Pipeline
from src.sysops.filesystem import _get_repo_root, setup_directories


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


def _hasbled_llm(index):
    data_dir = Path(_get_repo_root()) / "tests" / "data" / "reports_hasbled"
    stage1_file = data_dir / "reference_output.csv"
    df = pd.read_csv(stage1_file)
    return df[(df["index"] == index) & (df["llm_output"] != "False")]


def _hasbled_calc(index):
    data_dir = Path(_get_repo_root()) / "tests" / "data" / "reports_hasbled"
    stage1_file = data_dir / "reference_output.csv"
    df = pd.read_csv(stage1_file)
    return df[(df["index"] == index) & (df["llm_output"] == "False")]


@pytest.fixture
def hasbled_llm_P4ab463aa():
    return _hasbled_llm("P4ab463aa")


@pytest.fixture
def hasbled_llm_Pb4cf5ebc():
    return _hasbled_llm("Pb4cf5ebc")


@pytest.fixture
def hasbled_calc_P4ab463aa():
    return _hasbled_calc("P4ab463aa")


@pytest.fixture
def hasbled_calc_Pb4cf5ebc():
    return _hasbled_calc("Pb4cf5ebc")


@pytest.fixture
def hasbled_llm_Pb4cf5ebc():
    return _hasbled_llm("Pb4cf5ebc")


@pytest.fixture
def hasbled_calc_Pb4cf5ebc():
    return _hasbled_calc("Pb4cf5ebc")


def save_dict_as_yaml(d, file_path):
    with open(file_path, "w") as file:
        yaml.dump(d, file, sort_keys=False)


def try_numeric(col):
    try:
        return pd.to_numeric(col)
    except ValueError:
        return col


@pytest.mark.real_api
def test_checkpointing_llm_proceed_with_next_case1(
    default_cfg, hasbled_llm_P4ab463aa, hasbled_calc_P4ab463aa
):
    cfg = default_cfg.copy()
    # Test checkpointing with incompletely processed reports:
    # first report completed, second report not started yet
    score_str = cfg["risk_score"]
    # prepare env: store cfg in tmp dir
    cfg = default_cfg.copy()
    data_folder = Path(_get_repo_root()) / "tests/data/reports_hasbled"
    assert os.path.exists(data_folder)
    with tempfile.TemporaryDirectory() as tempdir:
        cfg_file = Path(tempdir) / "cfg.yaml"
        save_dict_as_yaml(cfg, cfg_file)

    # setup stage output dirs
    cfg_setup = cfg.copy()
    cfg_setup = setup_directories(cfg_setup)

    # store query extracts of text report
    df_llm = hasbled_llm_P4ab463aa.pivot(
        index="index", columns="item", values="value"
    ).reset_index()
    df_llm.columns.name = None
    results_file1 = Path(cfg_setup["stage1_dir"]) / Path(f"{score_str}_llm.csv")
    df_llm.to_csv(results_file1, index=False)

    # store calculated items of first report
    df_calc = hasbled_calc_P4ab463aa.pivot(
        index="index", columns="item", values="value"
    ).reset_index()
    df_calc.columns.name = None

    results_file2 = Path(cfg_setup["stage2_dir"]) / Path(f"{score_str}_calc.csv")
    df_calc.to_csv(results_file2, index=False)

    # test collected results as expected
    pipeline = Pipeline(cfg)
    results1 = pipeline._collect_results(
        results_file1, items=hasbled_llm_P4ab463aa["item"].to_list()
    )

    results1_tgt = hasbled_llm_P4ab463aa.pivot(
        index="index", columns="item", values="value"
    ).apply(try_numeric)
    results1_tgt.columns.name = None

    pdt.assert_frame_equal(results1_tgt, results1, check_like=True)

    # analogously for calculated results
    results2 = pipeline._collect_results(
        results_file2, items=hasbled_calc_P4ab463aa["item"].to_list()
    )
    results2_tgt = hasbled_calc_P4ab463aa.pivot(
        index="index", columns="item", values="value"
    ).apply(try_numeric)
    results2_tgt.columns.name = None
    results2_tgt = results2_tgt.reindex(columns=results2.columns)

    pdt.assert_frame_equal(results2_tgt, results2, check_like=True)


@pytest.mark.real_api
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
    with tempfile.TemporaryDirectory() as tempdir:
        cfg_file = Path(tempdir) / "cfg.yaml"
        save_dict_as_yaml(cfg, cfg_file)
        assert os.path.isfile(cfg_file)

        # setup stage output dirs
        cfg_setup = cfg.copy()
        cfg_setup = setup_directories(cfg_setup)

        # store query extracts of text report
        df_llm = hasbled_llm_P4ab463aa.pivot(
            index="index", columns="item", values="value"
        ).reset_index()
        df_llm.columns.name = None

        results_file1 = Path(cfg_setup["stage1_dir"]) / Path(f"{score_str}_llm.csv")
        df_llm.to_csv(results_file1, index=False)

        # store calculated items of first report
        df_calc = hasbled_calc_P4ab463aa.pivot(
            index="index", columns="item", values="value"
        ).reset_index()
        df_calc.columns.name = None
        results_file2 = Path(cfg_setup["stage2_dir"]) / Path(f"{score_str}_calc.csv")
        df_calc.to_csv(results_file2, index=False)

        # test collected results as expected
        pipeline = Pipeline(cfg)

        # call_per_item is patched, list of mocked_responses iterated instead
        mocked_responses = hasbled_llm_Pb4cf5ebc["llm_output"].to_list()

        with patch(
            "src.pipeline.provider_tools.Pipeline.call_per_item",
            side_effect=mocked_responses,
        ):
            async with Client(mcp_server) as client:
                _ = await client.call_tool(
                    "llm_pipeline",
                    {
                        "data_folder": data_folder,
                        "config_file": cfg_file,
                    },
                )

                # now test if checkpointed results_file1/2 have been augmented with case 2
                results1_post = pipeline._collect_results(
                    results_file1, items=hasbled_llm_P4ab463aa["item"].to_list()
                )
                results1_post_tgt = pd.concat(
                    [
                        hasbled_llm_P4ab463aa[["index", "item", "value"]],
                        hasbled_llm_Pb4cf5ebc[["index", "item", "value"]],
                    ]
                )
                results1_post_tgt = results1_post_tgt.pivot(
                    index="index", columns="item", values="value"
                )
                results1_post_tgt.columns.name = None

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

                pdt.assert_frame_equal(
                    results1_post_tgt, results1_post, check_like=True
                )

                # Now compare stage 2 results, ie, calculated items and final score
                results2_post = pipeline._collect_results(
                    results_file2, items=hasbled_calc_P4ab463aa["item"].to_list()
                )

                results2_post_tgt = pd.concat(
                    [hasbled_calc_P4ab463aa, hasbled_calc_Pb4cf5ebc],
                    axis=0,
                    ignore_index=True,
                )
                results2_post_tgt = results2_post_tgt.pivot(
                    index="index", columns="item", values="value"
                )
                results2_post_tgt = results2_post_tgt.reindex(
                    columns=results2_post.columns
                )
                # # harmonize stage 2 dataframes
                results2_post = results2_post.apply(try_numeric)
                results2_post_tgt = results2_post_tgt.apply(try_numeric)

                pdt.assert_frame_equal(
                    results2_post_tgt, results2_post, check_like=True, check_dtype=False
                )
