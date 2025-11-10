# tests/scoring/test_cha2ds2vasc.py

import pandas as pd
from pathlib import Path
import pytest
from unittest.mock import patch
import yaml

from src.scoring.cha2ds2vasc import CHA2DS2VAScScore


@pytest.fixture
def llm_row():
    return pd.Series(
        {
            "date_of_birth": "01.01.1970",  # A = A2 = 0
            "date_of_discharge": "01.01.2020",
            "is_female": 0,  # Sc = 0
            "chf_symptoms": 0,  # C = 0
            "chf_history": 0,  # C = 0
            "chf_lvef": 0,  # C = 0
            "hypertension_diagnosis": 0,  # H = 0
            "hypertension_medication": 0,  # H = 0
            "stroke_history": 0,  # S = 0
            "diabetes": 0,  # D = 0
            "hba1c": 6.4,  # D = 0
            "vascular_disease": 0,  # V = 0
        }
    )


@pytest.fixture
def calc_row():
    return pd.Series(
        {"C": 0, "H": 0, "A2": 0, "D": 0, "S2": 0, "V": 0, "A": 0, "Sc": 0, "score": 0}
    )


@pytest.fixture
def binary_variables():
    return [
        "is_female",
        "chf_symptoms",
        "chf_history",
        "chf_lvef",
        "hypertension_diagnosis",
        "hypertension_medication",
        "stroke_history",
        "diabetes",
        "vascular_disease",
    ]


def test_cha2ds2vasc_template_keys(llm_row):
    template_file = Path("src") / "prompts" / "cha2ds2vasc_template.yaml"
    with open(template_file) as f:
        template = yaml.safe_load(f)
    template_keys = set(template.keys())
    template_keys.remove("intro")
    llm_keys = set(llm_row.index)
    assert template_keys == llm_keys


def test_cha2ds2vasc_default(llm_row, calc_row):
    scorer = CHA2DS2VAScScore()
    output = scorer.calculate(llm_row)
    assert all(o == t for o, t in zip(output, calc_row))


def test_cha2ds2vasc_bool_type_true(llm_row, binary_variables):
    # test various boolean True representations
    src = llm_row.copy()
    scorer = CHA2DS2VAScScore()
    for b in ["YES", "True", "true", "TRUE", "1", "1.0", 1, True, "yes", "y"]:
        for var in binary_variables:
            print(f"var = {var}")
            src[var] = b
            output = scorer.calculate(src)
            if var == "stroke_history":
                assert output["score"] == 2
            else:
                assert output["score"] == 1
            src[var] = 0  # reset


def test_cha2ds2vasc_bool_type_false(llm_row, binary_variables):
    # test various boolean False representations
    src = llm_row.copy()
    scorer = CHA2DS2VAScScore()
    for b in ["NO", "False", "false", "FALSE", "0", "0.0", 0, False, "no", "n"]:
        for var in binary_variables:
            src[var] = b
            output = scorer.calculate(src)
            assert output["score"] == 0
            src[var] = 0  # reset


def test_cha2ds2vasc_invalid_age(llm_row):
    scorer = CHA2DS2VAScScore()
    src = llm_row.copy()
    for dob, dod in [("01.01.1800", "01.01.2020"), ("01.01.2020", "01.01.1800")]:
        src["date_of_birth"] = dob  # Will produce age > 150
        src["date_of_discharge"] = dod
        with patch.object(scorer.logger, "error") as mock_error:
            scorer.calculate(src)
            assert mock_error.called
            assert "Expected age in range [0:150]" in mock_error.call_args[0][0]


def test_cha2ds2vasc_age_65_74(llm_row, calc_row):
    # test age between 65 and 74
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = CHA2DS2VAScScore()
    for label, dob, dod, score in [
        (1, "01.01.1956", "01.01.2021", 1),  # A = 1
        (0, "01.01.1956", "31.12.2020", 0),  # A = 0
        (1, "01.01.1946", "01.01.2020", 1),  # A = 1
        (1, "02.01.1946", "01.01.2021", 1),  # A = 1
    ]:
        src["date_of_birth"] = dob
        src["date_of_discharge"] = dod
        tgt["A"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert all(o == t for o, t in zip(output, tgt))


def test_cha2ds2vasc_age_geq_75(llm_row, calc_row):
    # test age above 75
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = CHA2DS2VAScScore()
    for label, dob, dod, score in [
        (2, "01.01.1946", "01.01.2021", 2),  # A2 = 2, A = 1
        (0, "02.01.1946", "01.01.2021", 1),  # A2 = 0, A = 1
        (2, "01.01.1926", "01.01.2021", 2),  # A2 = 2, A = 1
    ]:
        src["date_of_birth"] = dob
        src["date_of_discharge"] = dod
        tgt["A2"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert output["A2"] == tgt["A2"]
        assert output["score"] == tgt["score"]


def test_cha2ds2vasc_hba1c_float_type(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = CHA2DS2VAScScore()
    for label, hba1c, score in [
        (0, 6.4, 0),
        (1, 6.5, 1),
        (0, "6.05", 0),
        (1, "6.5", 1),
        (1, "7", 1),
        (0, ".999", 0),
    ]:
        src["hba1c"] = hba1c
        tgt["D"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert output["D"] == tgt["D"]
        assert output["score"] == tgt["score"]


def test_cha2ds2vasc_diabetes_type(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = CHA2DS2VAScScore()
    for label, diabetes, hbac1, score in [
        (0, 0, 6.4, 0),
        (1, 0, 6.5, 1),
        (1, 1, 6.5, 1),
        (1, 1, 5.5, 1),
    ]:
        src["diabetes"] = label
        src["hba1c"] = hbac1
        tgt["D"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert output["D"] == tgt["D"]
        assert output["score"] == tgt["score"]


def test_cha2ds2vasc_stroke(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = CHA2DS2VAScScore()
    for label, stroke_history, score in [
        (0, 0, 0),
        (2, 1, 2),
    ]:
        src["stroke_history"] = stroke_history
        tgt["S2"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert output["S2"] == tgt["S2"]
        assert output["score"] == tgt["score"]


def test_cha2ds2vasc_vascular_disease(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = CHA2DS2VAScScore()
    for label, vascular_disease, score in [
        (0, 0, 0),
        (1, 1, 1),
    ]:
        src["vascular_disease"] = vascular_disease
        tgt["V"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert output["V"] == tgt["V"]
        assert output["score"] == tgt["score"]


def test_cha2ds2vasc_sex_category(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = CHA2DS2VAScScore()
    for label, is_female, score in [
        (0, 0, 0),
        (1, 1, 1),
    ]:
        src["is_female"] = is_female
        tgt["Sc"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert output["Sc"] == tgt["Sc"]
        assert output["score"] == tgt["score"]
