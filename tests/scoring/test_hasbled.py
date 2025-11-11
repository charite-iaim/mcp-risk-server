# /tests/scoring/test_hasbled.py

import pandas as pd
from pathlib import Path
import pytest
import yaml

from src.scoring.hasbled import HASBLEDScore


@pytest.fixture
def llm_row():
    return pd.Series(
        {
            "sys_bp": "105",  # H = 0
            "renal_disease": False,
            "creatinine": "2.2",  # A1 = 0
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
def calc_row():
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

def test_hasbled_template_keys(llm_row):
    template_file = Path("src") / "prompts" / "hasbled_template.yaml" 
    with open(template_file) as f:
        template = yaml.safe_load(f)
    template_keys = set(template.keys())
    template_keys.remove("intro")
    llm_keys = set(llm_row.index)
    assert template_keys == llm_keys


def test_hasbled_default(llm_row, calc_row):
    scorer = HASBLEDScore()
    output = scorer.calculate(llm_row)
    assert all(o == t for o, t in zip(output, calc_row))


def test_hasbled_bool_type_true(llm_row, calc_row):
    # test various boolean True representations:
    #  ["YES", "True", "true", "TRUE", "1", "1.0", 1, True, "yes", "y"]
    src = llm_row.copy()
    target = calc_row.copy()
    scorer = HASBLEDScore()
    src["unit_mg_per_dL"] = "YES"
    src["renal_disease"] = "True"  # A1 = 1
    target["A1"] = 1
    target["score"] = 1
    output = scorer.calculate(src)
    assert all(o == t for o, t in zip(output, target))

    src["liver_disease"] = "yes"  # A2 = 1
    target["A2"] = 1
    target["score"] = 2
    output = scorer.calculate(src)
    assert all(o == t for o, t in zip(output, target))

    src["stroke_history"] = "true"  # S = 1
    target["S"] = 1
    target["score"] = 3
    output = scorer.calculate(src)
    assert all(o == t for o, t in zip(output, target))

    src["bleeding_history"] = "1"  # B = 1
    src["is_female"] = 1
    target["B"] = 1
    target["score"] = 4
    output = scorer.calculate(src)
    assert all(o == t for o, t in zip(output, target))

    src["labile_inr"] = "y"  # L = 1
    target["L"] = 1
    target["score"] = 5
    output = scorer.calculate(src)
    assert all(o == t for o, t in zip(output, target))

    src["med_bleed"] = True  # D1 = 1
    target["D1"] = 1
    target["score"] = 6
    output = scorer.calculate(src)
    assert all(o == t for o, t in zip(output, target))

    src["alcohol"] = "TRUE"  # D2 = 1
    target["D2"] = 1
    target["score"] = 7
    output = scorer.calculate(src)
    assert all(o == t for o, t in zip(output, target))


def test_hasbled_bool_type_false(llm_row, calc_row):
    # test various boolean True representations:
    #  ["False", "false", "FALSE", "0", "0.0", 0, False, "no", "n"]
    src = llm_row.copy()
    target = calc_row.copy()
    src["renal_disease"] = "False"
    src["liver_disease"] = "false"
    src["stroke_history"] = "FALSE"
    src["bleeding_history"] = "0"
    src["is_female"] = 0
    src["labile_inr"] = False
    src["med_bleed"] = "no"
    src["alcohol"] = "n"
    scorer = HASBLEDScore()
    output = scorer.calculate(src)
    assert all(o == t for o, t in zip(output, target))


def test_hasbled_float_type(llm_row, calc_row):
    # test various float type representation
    src = llm_row.copy()
    target = calc_row.copy()
    src["sys_bp"] = 105  # H = 0
    src["renal_disease"] = 0.0
    src["creatinine"] = "2.2"  # A1 = 0
    src["ast"] = "140"
    src["bilirubin"] = 2.4  # A2 = 0
    src["alp"] = ".43"
    src["alt"] = "130.3333333"
    scorer = HASBLEDScore()
    output = scorer.calculate(src)
    assert all(o == t for o, t in zip(output, target))


def test_hypertension(llm_row, calc_row):
    # test hypertension
    src = llm_row.copy()
    target = calc_row.copy()
    target["H"] = 1
    target["score"] = 1
    src["sys_bp"] = 161  # H = 1
    scorer = HASBLEDScore()
    output = scorer.calculate(src)
    assert all(o == t for o, t in zip(output, target))


def test_abnormal_liver_function_0(llm_row, calc_row):
    src = llm_row.copy()
    target = calc_row.copy()
    src["liver_disease"] = True  # A2 = 1
    target["A2"] = 1
    target["score"] = 1
    scorer = HASBLEDScore()
    output = scorer.calculate(src)
    assert all(o == t for o, t in zip(output, target))


def test_abnormal_liver_function_1(llm_row, calc_row):
    # test abnormal liver function
    # point for bilirubin as ULN missing
    src = llm_row.copy()
    target = calc_row.copy()
    ast_no_point = ["missing", 3 * 35 - 1]
    alt_no_point = ["missing", 3 * 33 - 1]
    alp_no_point = ["missing", 3 * 130 - 1]
    for bilirubin, score in zip(["2.5", "2.6", "2.7", "3"], [0, 0, 1, 1]):
        src["bilirubin"] = bilirubin
        for ast in ast_no_point:
            src["ast"] = ast
            for alt in alt_no_point:
                src["alt"] = alt
                for alp in alp_no_point:
                    src["alp"] = alp
                    scorer = HASBLEDScore()
                    output = scorer.calculate(src)
                    if all(x == "missing" for x in [ast, alt, alp]):
                        target["score"] = score
                    else:
                        target["score"] = 0
                    target["A2"] = target["score"]
                    assert all(o == t for o, t in zip(output, target))


def test_abnormal_liver_function_2(llm_row, calc_row):
    # test abnormal liver function for AST
    src = llm_row.copy()
    target = calc_row.copy()
    src["bilirubin"] = "2.7"
    alt_no_point = ["missing", 3 * 33 - 1]
    alp_no_point = ["missing", 3 * 130 - 1]
    for ast, score in zip([100, 105], [0, 1]):
        src["ast"] = ast
        for alt in alt_no_point:
            src["alt"] = alt
            for alp in alp_no_point:
                src["alp"] = alp
                scorer = HASBLEDScore()
                output = scorer.calculate(src)
                target["A2"] = score
                target["score"] = score
                assert all(o == t for o, t in zip(output, target))


def test_abnormal_liver_function_3(llm_row, calc_row):
    # test abnormal liver function for ALT
    src = llm_row.copy()
    target = calc_row.copy()
    src["bilirubin"] = "2.7"
    ast_no_point = ["missing", 3 * 35 - 1]
    alp_no_point = ["missing", 3 * 130 - 1]
    alt_list = [3 * 33 - 1, 3 * 33, 3 * 45 - 1, 3 * 45]
    is_female_list = [1, 1, 0, 0]
    score_list = [0, 1, 0, 1]
    for alt, is_female, score in zip(alt_list, is_female_list, score_list):
        src["alt"] = alt
        src["is_female"] = is_female
        for ast in ast_no_point:
            src["ast"] = ast
            for alp in alp_no_point:
                src["alp"] = alp
                scorer = HASBLEDScore()
                output = scorer.calculate(src)
                target["A2"] = score
                target["score"] = score
                assert all(o == t for o, t in zip(output, target))


def test_abnormal_liver_function_4(llm_row, calc_row):
    # test abnormal liver function for ALP
    src = llm_row.copy()
    target = calc_row.copy()
    src["bilirubin"] = "2.7"
    ast_no_point = ["missing", 3 * 35 - 1]
    alt_no_point = ["missing", 3 * 33 - 1, 3 * 45 - 1]
    is_female_list = [1, 1, 0]
    alt_female_list = zip(alt_no_point, is_female_list)
    alp_list = [3 * 130 - 1, 3 * 130]
    score_list = [0, 1]
    for alp, score in zip(alp_list, score_list):
        src["alp"] = alp
        for ast in ast_no_point:
            src["ast"] = ast
            for alt, is_female in alt_female_list:
                src["alt"] = alt
                src["is_female"] = is_female
                scorer = HASBLEDScore()
                output = scorer.calculate(src)
                target["A2"] = score
                target["score"] = score
                assert all(o == t for o, t in zip(output, target))


def test_bleeding_risk(llm_row, calc_row):
    # test bleeding risk
    src = llm_row.copy()
    target = calc_row.copy()
    scorer = HASBLEDScore()

    src["hemoglobin"] = 12  # B = 1
    src["is_female"] = 0
    output = scorer.calculate(src)
    target["B"] = 1
    target["score"] = 1
    assert all(o == t for o, t in zip(output, target))

    src["is_female"] = 1  # B = 0
    output = scorer.calculate(src)
    target["B"] = 0
    target["score"] = 0
    assert all(o == t for o, t in zip(output, target))

    src["hemoglobin"] = 11.9  # B = 1
    output = scorer.calculate(src)
    target["B"] = 1
    target["score"] = 1
    assert all(o == t for o, t in zip(output, target))

    src["hemoglobin"] = 14.9999  # B = 0
    src["is_female"] = 0
    output = scorer.calculate(src)
    target["B"] = 0
    target["score"] = 0
    assert all(o == t for o, t in zip(output, target))


def test_hasbled_labile_inr_true(llm_row, calc_row):
    # test labile INR
    src = llm_row.copy()
    target = calc_row.copy()
    src["labile_inr"] = True  # L = 1
    scorer = HASBLEDScore()
    output = scorer.calculate(src)
    target["L"] = 1
    target["score"] = 1
    assert all(o == t for o, t in zip(output, target))


def test_hasbled_elderly_true(llm_row, calc_row):
    # test various float type representation
    src = llm_row.copy()
    target = calc_row.copy()
    scorer = HASBLEDScore()

    src["date_of_birth"] = "31.12.1954"
    src["date_of_discharge"] = "01.01.2020"  # E = 1
    output = scorer.calculate(src)
    target["E"] = 1
    target["score"] = 1
    assert all(o == t for o, t in zip(output, target))

    src["date_of_birth"] = "01.01.1955"  # E = 0
    output = scorer.calculate(src)
    target["E"] = 0
    target["score"] = 0
    assert all(o == t for o, t in zip(output, target))


def test_drugs_true(llm_row, calc_row):
    # test drugs
    src = llm_row.copy()
    target = calc_row.copy()
    scorer = HASBLEDScore()

    # D1 = True, D2 = False
    src["med_bleed"] = True  # D1 = 1, D2 = 0
    output = scorer.calculate(src)
    target["D1"] = 1
    target["score"] = 1
    assert all(o == t for o, t in zip(output, target))

    src["alcohol"] = True  # D1 = 1, D2 = 1
    output = scorer.calculate(src)
    target["D2"] = 1
    target["score"] = 2
    assert all(o == t for o, t in zip(output, target))

    src["med_bleed"] = False  # D1 = 0, D2 = 1
    output = scorer.calculate(src)
    target["D1"] = 0
    target["score"] = 1
    assert all(o == t for o, t in zip(output, target))

