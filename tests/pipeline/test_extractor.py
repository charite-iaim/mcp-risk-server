# /tests/pipeline/test_extractor.py

from pathlib import Path
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from src.pipeline.provider_tools import Extractor


@pytest.fixture
def hasbled_completions():
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
    }


def test_hasbled_extraction(hasbled_completions):

    mocked_responses = [v[0] for v in hasbled_completions.values()]
    values = [v[1] for v in hasbled_completions.values()]
    extractor = Extractor()
    for item, values in hasbled_completions.items():
        response, value_extracted = values
        value_dict = extractor(response)
        assert len(value_dict) == 1
        assert item in value_dict
        assert value_dict.get(item) == value_extracted
