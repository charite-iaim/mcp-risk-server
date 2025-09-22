# /tests/pipeline/test_call_calc.py

import pandas as pd
from pathlib import Path
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from src.pipeline.provider_tools import Pipeline


@pytest.fixture
def default_llm_hasbled():
    return pd.Series(
        {
            "sys_bp": "105",  # H = 0
            "renal_disease": False,
            "dialisis": False,
            "creatine": "2.2",  # A1 = 0
            "unit_mg_per_dL": True,
            "liver_disease": False,
            "ast": "missing",
            "alt": "missing",
            "alp": "missing",
            "bilirubin": "2.4",  # A2 = 0
            "stroke_history": False,  # S = 0
            "bleeding_history": "0",
            "is_female": 0,
            "hemoglobin": "15",  # B = 0
            "labile_inr": False,  # L = 0
            "date_of_birth": "01.01.1970",
            "date_of_discharge": "01.01.2020",  # E = 0
            "med_bleed": False,  # D1 = 0
            "alcohol": False,  # D2 = 0
        }
    )


@pytest.fixture
def default_calc_hasbled():
    return pd.Series(
        {
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
def default_config():
    with tempfile.TemporaryDirectory() as tempdir:
        return {
            "run_name": "test_run",
            "risk_score": "hasbled",
            "provider": "deepseek",
            "network": {"http_proxy": None, "https_proxy": None},
            "model": "mock-model",
            "output_folder": Path(tempdir) / Path("outputs"),
            "api_key": "mock-api-key",
            "org_key": None,
            "project_id": None,
        }


def test_call_calc_hasbled(default_config, default_llm_hasbled):
    results_llm = default_llm_hasbled.copy()
    text_id = "test_id"
    config = default_config.copy()
    pipeline = Pipeline(config)
    results_calc = pipeline.call_calc(results_llm, text_id)
    assert results_calc["score"] == 0


def make_pipeline_and_row(config, score_str):
    config = config.copy()
    config["risk_score"] = score_str
    pipeline = Pipeline(config)
    # Minimal DataFrame for results_row
    results_row = pd.DataFrame({"score": [1], "bp": [1]})
    return pipeline, results_row


def test_call_calc_results_file2_not_exists(default_config):
    score_str = "hasbled"
    text_id = "test_id"
    pipeline, results_row = make_pipeline_and_row(default_config, score_str)

    with patch("pathlib.Path.exists", return_value=False), patch(
        "pandas.DataFrame", wraps=pd.DataFrame
    ) as mock_df, patch("pandas.read_csv") as mock_read_csv, patch(
        "src.scoring.base.RiskScoreFactory.create"
    ) as mock_factory:

        # Return a real Series for score_items
        score_items = pd.Series([1, 2], index=["score", "item1"])
        mock_score = MagicMock()
        mock_score.calculate.return_value = score_items
        mock_factory.return_value = mock_score

        pipeline.call_calc(results_row, text_id)
        mock_df.assert_called()  # DataFrame should be created
        mock_read_csv.assert_not_called()  # read_csv should not be called


def test_call_calc_results_file2_exists(default_config, default_llm_hasbled):
    score_str = "hasbled"
    text_id = "test_id"
    config = default_config.copy()
    config["risk_score"] = score_str
    pipeline = Pipeline(config)

    # Mock Path.exists to return True (file exists)
    with patch.object(type(pipeline.results_file2), "exists", return_value=True):
        # Mock pd.read_csv to simulate loading
        with patch(
            "pandas.read_csv",
            return_value=pd.DataFrame({"index": [0], "score": [0]}),
        ) as mock_read_csv:
            # Mock pd.DataFrame to ensure it's not called
            with patch("pandas.DataFrame") as mock_df:
                pipeline.call_calc(default_llm_hasbled.copy(), text_id)
                mock_read_csv.assert_called_once_with(
                    pipeline.results_file2, index_col="index"
                )
                mock_df.assert_not_called()  # DataFrame should not be created
