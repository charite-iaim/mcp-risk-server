# tests/pipeline/test_call_pipeline.py
import pytest
import tempfile
from unittest.mock import patch, MagicMock
import yaml

from src.pipeline.provider_tools import _llm_pipeline_inner


@pytest.fixture
def default_config():
    return {
        "run_name": "test_run",
        "risk_score": "happyness",
        "provider": "mock-provider",
        "network": {"http_proxy": None, "https_proxy": None},
        "model": "mock-model",
        "output_dir": "./output",
        "api": {
            "api_key": "mock-api-key",
            "org_key": "mock-org-key",
            "project_id": "mock-project-id",
        },
        "org_key": "mock-org-key",
        "project_id": "mock-project-id",
    }


@pytest.fixture
def mock_pipeline():
    with patch("src.pipeline.provider_tools.Pipeline") as MockPipeline:
        instance = MockPipeline.return_value
        instance.call_llm.return_value = {"item1": 1, "item2": 2}
        instance.call_calc.return_value = 42
        yield MockPipeline


@pytest.fixture
def mocks(default_config):
    with patch(
        "src.pipeline.provider_tools.get_score_str", return_value="HASBLED"
    ) as mock_get_score_str, patch(
        "src.pipeline.provider_tools.os.path.isfile", return_value=True
    ) as mock_isfile, patch(
        "src.pipeline.provider_tools.open", create=True
    ) as mock_open, patch(
        "src.pipeline.provider_tools.yaml.safe_load", return_value=default_config
    ) as mock_yaml, patch(
        "src.pipeline.provider_tools.read_text_files",
        return_value={"case1": "Patient text"},
    ) as mock_read_text_files:
        yield {
            "get_score_str": mock_get_score_str,
            "isfile": mock_isfile,
            "open": mock_open,
            "yaml": mock_yaml,
            "read_text_files": mock_read_text_files,
        }


@pytest.fixture
def temp_config_file(default_config):
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".yaml") as f:
        yaml.dump(default_config, f)
        yield f.name


def test_llm_pipeline_happy_path(mocks, mock_pipeline, temp_config_file):
    instance = mock_pipeline.return_value
    instance.call_llm.return_value = "some llm output"
    instance.call_calc.return_value = {"score": 7.2}

    result = _llm_pipeline_inner("dummy_folder", temp_config_file)

    mock_pipeline.assert_called_once()
    instance.call_llm.assert_called()
    instance.call_calc.assert_called()


def test_llm_pipeline_config_file_missing(mocks):
    mocks["isfile"].return_value = False
    with pytest.raises(AssertionError):
        _llm_pipeline_inner("dummy_folder", "dummy_config.yaml")


def test_llm_pipeline_empty_data(mocks, mock_pipeline):
    mocks["read_text_files"].return_value = {}
    result = _llm_pipeline_inner("dummy_folder", "dummy_config.yaml")
    # Should not call pipeline methods if no data
    instance = mock_pipeline.return_value
    instance.call_llm.assert_not_called()
    instance.call_calc.assert_not_called()
