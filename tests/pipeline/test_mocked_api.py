# src/tests/pipeline/test_mocked_api.py

import pytest
from unittest.mock import patch, MagicMock

from src.pipeline.provider_tools import Pipeline


@pytest.fixture
def default_config():
    return {
        "run_name": "test_run",
        "risk_score": "happyness",
        "provider": "mock",
        "network": {"http_proxy": None, "https_proxy": None},
        "model": "mock-model",
        "output_folder": "./outputs",
        "api_key": "mock-api-key",
        "org_key": None,
        "project_id": None,
    }


@pytest.mark.mock_api
def test_pipeline_raises_on_unknown_provider(default_config):
    with pytest.raises(ValueError, match="Unknown API"):
        Pipeline(default_config)


@pytest.mark.mock_api
def test_pipeline_init(default_config):
    config = default_config.copy()
    config["provider"] = "openai"
    pipeline = Pipeline(config)
    assert pipeline is not None


@pytest.mark.mock_api
def test_pipeline_config_attributes(default_config):
    config = default_config.copy()
    config["provider"] = "openai"
    pipeline = Pipeline(config)
    assert pipeline._cfg["run_name"] == "test_run"
    assert pipeline._cfg["provider"] == "openai"
    assert pipeline._cfg["model"] == "mock-model"
    assert pipeline._cfg["output_folder"] == "./outputs"
    assert pipeline._cfg["api_key"] == "mock-api-key"


@pytest.mark.mock_api
def test_api_call_per_item(default_config):
    config = default_config.copy()
    config["provider"] = "openai"
    pipeline = Pipeline(config)
    mock_choice = MagicMock()
    mock_choice.message.content = "Mock response"
    # Mock the API call
    pipeline.client.chat.completions.create = MagicMock(
        return_value=MagicMock(choices=[mock_choice])
    )
    response = pipeline.call_per_item("Test prompt.", "Be concise.")
    assert response == "Mock response"
