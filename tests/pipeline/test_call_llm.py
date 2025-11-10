# /tests/pipeline/test_call_llm.py

from datetime import datetime
import os
from pandas import concat, read_csv, DataFrame, Series
import pandas.testing as pdt
from pathlib import Path
import pytest
import tempfile
from unittest.mock import patch, MagicMock
import yaml

from src.pipeline.provider_tools import Pipeline
from src.sysops.names import get_score_str


@pytest.fixture
def default_cfg():
    with tempfile.TemporaryDirectory() as tempdir:
        return {
            "run_name": "test_run",
            "risk_score": "hasbled",
            "provider": "deepseek",
            "network": {"http_proxy": None, "https_proxy": None},
            "model": "mock-model",
            "output_dir": Path(tempdir) / Path("output"),
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


def save_dict_as_yaml(d, file_path):
    with open(file_path, "w") as file:
        yaml.dump(d, file, sort_keys=False)


# def streaming_mock(*args, **kwargs):
#     # Simulate an iterator over "chunks"
#     for text in ["my", "_", "mocked", "_", "string"]:
#         mock_chunk = MagicMock()
#         mock_delta = MagicMock()
#         mock_delta.content = text
#         mock_chunk.choices = [MagicMock(delta=mock_delta)]
#         yield mock_chunk

# with patch("path.to.your.instance.client.chat.completions.create", side_effect=streaming_mock):
#     result = instance.call_per_item("input prompt", "system prompt")
#     assert result == "my_mocked_string"


def test_call_per_item(default_cfg):
    cfg = default_cfg.copy()
    pipeline = Pipeline(cfg)
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Patient has no congestive heart failure {C: 0}"
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    with patch.object(
        pipeline.client.chat.completions, "create", return_value=mock_completion
    ):
        result = pipeline.call_per_item("prompt", "system prompt")
        assert result == mock_message.content


def test_collect_results_init(default_cfg, hasbled_llm_P4ab463aa):
    from src.sysops.names import get_score_str

    cfg = default_cfg.copy()
    pipeline = Pipeline(cfg)
    score_str = get_score_str(cfg["risk_score"])
    with tempfile.TemporaryDirectory() as tempdir:
        results_file1 = Path(tempdir) / Path(f"{score_str}_llm.csv")
        items = hasbled_llm_P4ab463aa.keys()
        results = pipeline._collect_results(results_file1, items)
        assert results.empty
        assert set(results.columns) == set(hasbled_llm_P4ab463aa.keys())


def test_checkpointing_llm_init(default_cfg, hasbled_llm_P4ab463aa):
    import os

    # Test checkpoint folder creation for initial run
    # mock client.chat.completions.create called by call_llm -> call_per_item
    # two logs should be written under log dir / <item>_<ts>.log
    cfg = default_cfg.copy()
    pipeline = Pipeline(cfg)
    score_str = get_score_str(cfg["risk_score"])
    text_id = "Case_12345"
    mocked_responses = [v[0] for v in hasbled_llm_P4ab463aa.values()]
    solution_series = Series({k: v[1] for k, v in hasbled_llm_P4ab463aa.items()})
    solution_series.name = text_id
    solution_df = DataFrame({k: [v[1]] for k, v in hasbled_llm_P4ab463aa.items()})
    solution_df["index"] = [text_id]
    results_file1 = pipeline.stage1_dir / Path(f"{score_str}_llm.csv")
    log_dir = pipeline.log_base_dir / text_id
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # call_per_item is patched, list of mocked_responses iterated instead
    with patch.object(pipeline, "call_per_item", side_effect=mocked_responses):
        target_series = pipeline.call_llm(text="Report text", text_id=text_id, ts=ts)
        assert solution_series.shape[0] == target_series.shape[0]
        # check results row
        pdt.assert_series_equal(target_series, solution_series)
        # check existence of results file
        assert os.path.exists(results_file1)
        target_from_file = read_csv(results_file1)
        pdt.assert_frame_equal(solution_df, target_from_file, check_like=True)
        # check existence of log file
        assert os.path.exists(log_dir)
        for item, values in hasbled_llm_P4ab463aa.items():
            response = values[0]
            log_file = log_dir / Path(f"{item}_{ts}.log")
            assert os.path.exists(log_file)
            with open(log_file) as f:
                log = f.read()
                assert response in log


def test_checkpointing_llm_proceed_within_case(default_cfg, hasbled_llm_P4ab463aa):
    # Test checkpointing with a single incomplete processed report
    # Assume items[0:12] have already been processed and backed up
    cfg = default_cfg.copy()
    pipeline = Pipeline(cfg)
    score_str = get_score_str(cfg["risk_score"])
    text_id = "Case_12345"
    idx = 12  # item index until which queries have already been processed
    mocked_responses = [v[0] for v in list(hasbled_llm_P4ab463aa.values())[idx:]]
    # write back to results file first idx_done items
    results_file1 = pipeline.stage1_dir / Path(f"{score_str}_llm.csv")

    solution_part1_df = DataFrame(
        {k: [v[1]] for k, v in list(hasbled_llm_P4ab463aa.items())[:idx]}
    )
    solution_part1_df["index"] = [text_id]
    solution_part1_df.to_csv(results_file1, index=False)

    # 2nd part will be retrieved in pipeline call and complemented
    solution_part2_df = DataFrame(
        {k: [v[1]] for k, v in list(hasbled_llm_P4ab463aa.items())[idx:]}
    )

    solution_df = concat([solution_part1_df, solution_part2_df], axis=1)

    solution_part2_df["index"] = [text_id]

    # return of call() will be complete list of items built from
    # checkpointed and call extracted item values
    solution_series = Series({k: v[1] for k, v in hasbled_llm_P4ab463aa.items()})
    solution_series.name = text_id

    log_dir = pipeline.log_base_dir / text_id
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # call_per_item is patched, list of mocked_responses iterated instead
    with patch.object(pipeline, "call_per_item", side_effect=mocked_responses):
        target_series = pipeline.call_llm(text="Report text", text_id=text_id, ts=ts)

        assert solution_series.shape[0] == target_series.shape[0]
        # check results row
        pdt.assert_series_equal(target_series, solution_series)
        # check existence of results file
        assert os.path.exists(results_file1)
        target_from_file = read_csv(results_file1)

        pdt.assert_frame_equal(solution_df, target_from_file, check_like=True)
        # check existence of log file
        assert os.path.exists(log_dir)
        for item, values in list(hasbled_llm_P4ab463aa.items())[idx:]:
            response = values[0]
            log_file = log_dir / Path(f"{item}_{ts}.log")
            assert os.path.exists(log_file)
            with open(log_file) as f:
                log = f.read()
                assert response in log
