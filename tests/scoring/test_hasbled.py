# /tests/scoring/test_hasbled.py

import pandas as pd
import pytest


from src.scoring.hasbled import HASBLEDScore


@pytest.fixture
def default_row():
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
def default_target():
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


def test_hasbled_score_default(default_row, default_target):
    scorer = HASBLEDScore()
    output = scorer.calculate(default_row)
    assert all(o == t for o, t in zip(output, default_target))


def test_hasbled_score_bool_type_true(default_row, default_target):
    # test various boolean True representations:
    #  ["YES", "True", "true", "TRUE", "1", "1.0", 1, True, "yes", "y"]
    s = default_row.copy()
    target = default_target.copy()
    scorer = HASBLEDScore()
    s["unit_mg_per_dL"] = "YES"
    s["dialysis"] = "1.0"
    s["renal_disease"] = "True"  # A1 = 1
    target["A1"] = 1
    target["score"] = 1
    output = scorer.calculate(s)
    assert all(o == t for o, t in zip(output, target))

    s["liver_disease"] = "yes"  # A2 = 1
    target["A2"] = 1
    target["score"] = 2
    output = scorer.calculate(s)
    assert all(o == t for o, t in zip(output, target))

    s["stroke_history"] = "true"  # S = 1
    target["S"] = 1
    target["score"] = 3
    output = scorer.calculate(s)
    assert all(o == t for o, t in zip(output, target))

    s["bleeding_history"] = "1"  # B = 1
    s["is_female"] = 1
    target["B"] = 1
    target["score"] = 4
    output = scorer.calculate(s)
    assert all(o == t for o, t in zip(output, target))

    s["labile_inr"] = "y"  # L = 1
    target["L"] = 1
    target["score"] = 5
    output = scorer.calculate(s)
    assert all(o == t for o, t in zip(output, target))

    s["med_bleed"] = True  # D1 = 1
    target["D1"] = 1
    target["score"] = 6
    output = scorer.calculate(s)
    assert all(o == t for o, t in zip(output, target))

    s["alcohol"] = "TRUE"  # D2 = 1
    target["D2"] = 1
    target["score"] = 7
    output = scorer.calculate(s)
    assert all(o == t for o, t in zip(output, target))


def test_hasbled_score_bool_type_false(default_row, default_target):
    # test various boolean True representations:
    #  ["False", "false", "FALSE", "0", "0.0", 0, False, "no", "n"]
    s = default_row.copy()
    target = default_target.copy()
    s["renal_disease"] = "False"
    s["dialysis"] = "0.0"
    s["liver_disease"] = "false"
    s["stroke_history"] = "FALSE"
    s["bleeding_history"] = "0"
    s["is_female"] = 0
    s["labile_inr"] = False
    s["med_bleed"] = "no"
    s["alcohol"] = "n"
    scorer = HASBLEDScore()
    output = scorer.calculate(s)
    assert all(o == t for o, t in zip(output, target))


def test_hasbled_score_float_type(default_row, default_target):
    # test various float type representation
    s = default_row.copy()
    target = default_target.copy()
    s["sys_bp"] = 105  # H = 0
    s["renal_disease"] = 0.0
    s["creatine"] = "2.2"  # A1 = 0
    s["ast"] = "140"
    s["bilirubin"] = 2.4  # A2 = 0
    s["alp"] = ".43"
    s["alt"] = "130.3333333"
    scorer = HASBLEDScore()
    output = scorer.calculate(s)
    assert all(o == t for o, t in zip(output, target))


def test_hypertension(default_row, default_target):
    # test hypertension
    s = default_row.copy()
    target = default_target.copy()
    target["H"] = 1
    target["score"] = 1
    s["sys_bp"] = 161  # H = 1
    scorer = HASBLEDScore()
    output = scorer.calculate(s)
    assert all(o == t for o, t in zip(output, target))


def test_abnormal_liver_function_0(default_row, default_target):
    s = default_row.copy()
    target = default_target.copy()
    s["liver_disease"] = True  # A2 = 1
    target["A2"] = 1
    target["score"] = 1
    scorer = HASBLEDScore()
    output = scorer.calculate(s)
    assert all(o == t for o, t in zip(output, target))


def test_abnormal_liver_function_1(default_row, default_target):
    # test abnormal liver function
    # point for bilirubin as ULN missing
    s = default_row.copy()
    target = default_target.copy()
    ast_no_point = ["missing", 3 * 35 - 1]
    alt_no_point = ["missing", 3 * 33 - 1]
    alp_no_point = ["missing", 3 * 130 - 1]
    for bilirubin, score in zip(["2.5", "2.6", "2.7", "3"], [0, 0, 1, 1]):
        s["bilirubin"] = bilirubin
        for ast in ast_no_point:
            s["ast"] = ast
            for alt in alt_no_point:
                s["alt"] = alt
                for alp in alp_no_point:
                    s["alp"] = alp
                    scorer = HASBLEDScore()
                    output = scorer.calculate(s)
                    if all(x == "missing" for x in [ast, alt, alp]):
                        target["score"] = score
                    else:
                        target["score"] = 0
                    target["A2"] = target["score"]
                    assert all(o == t for o, t in zip(output, target))


def test_abnormal_liver_function_2(default_row, default_target):
    # test abnormal liver function for AST
    s = default_row.copy()
    target = default_target.copy()
    s["bilirubin"] = "2.7"
    alt_no_point = ["missing", 3 * 33 - 1]
    alp_no_point = ["missing", 3 * 130 - 1]
    for ast, score in zip([100, 105], [0, 1]):
        s["ast"] = ast
        for alt in alt_no_point:
            s["alt"] = alt
            for alp in alp_no_point:
                s["alp"] = alp
                scorer = HASBLEDScore()
                output = scorer.calculate(s)
                target["A2"] = score
                target["score"] = score
                assert all(o == t for o, t in zip(output, target))


def test_abnormal_liver_function_3(default_row, default_target):
    # test abnormal liver function for ALT
    s = default_row.copy()
    target = default_target.copy()
    s["bilirubin"] = "2.7"
    ast_no_point = ["missing", 3 * 35 - 1]
    alp_no_point = ["missing", 3 * 130 - 1]
    alt_list = [3 * 33 - 1, 3 * 33, 3 * 45 - 1, 3 * 45]
    is_female_list = [1, 1, 0, 0]
    score_list = [0, 1, 0, 1]
    for alt, is_female, score in zip(alt_list, is_female_list, score_list):
        s["alt"] = alt
        s["is_female"] = is_female
        for ast in ast_no_point:
            s["ast"] = ast
            for alp in alp_no_point:
                s["alp"] = alp
                scorer = HASBLEDScore()
                output = scorer.calculate(s)
                target["A2"] = score
                target["score"] = score
                assert all(o == t for o, t in zip(output, target))


def test_abnormal_liver_function_4(default_row, default_target):
    # test abnormal liver function for ALP
    s = default_row.copy()
    target = default_target.copy()
    s["bilirubin"] = "2.7"
    ast_no_point = ["missing", 3 * 35 - 1]
    alt_no_point = ["missing", 3 * 33 - 1, 3 * 45 - 1]
    is_female_list = [1, 1, 0]
    alt_female_list = zip(alt_no_point, is_female_list)
    alp_list = [3 * 130 - 1, 3 * 130]
    score_list = [0, 1]
    for alp, score in zip(alp_list, score_list):
        s["alp"] = alp
        for ast in ast_no_point:
            s["ast"] = ast
            for alt, is_female in alt_female_list:
                s["alt"] = alt
                s["is_female"] = is_female
                scorer = HASBLEDScore()
                output = scorer.calculate(s)
                target["A2"] = score
                target["score"] = score
                assert all(o == t for o, t in zip(output, target))


def test_bleeding_risk(default_row, default_target):
    # test bleeding risk
    s = default_row.copy()
    target = default_target.copy()
    scorer = HASBLEDScore()

    s["hemoglobin"] = 12  # B = 1
    s["is_female"] = 0
    output = scorer.calculate(s)
    target["B"] = 1
    target["score"] = 1
    assert all(o == t for o, t in zip(output, target))

    s["is_female"] = 1  # B = 0
    output = scorer.calculate(s)
    target["B"] = 0
    target["score"] = 0
    assert all(o == t for o, t in zip(output, target))

    s["hemoglobin"] = 11.9  # B = 1
    output = scorer.calculate(s)
    target["B"] = 1
    target["score"] = 1
    assert all(o == t for o, t in zip(output, target))

    s["hemoglobin"] = 14.9999  # B = 0
    s["is_female"] = 0
    output = scorer.calculate(s)
    target["B"] = 0
    target["score"] = 0
    assert all(o == t for o, t in zip(output, target))


def test_hasbled_score_labile_inr_true(default_row, default_target):
    # test labile INR
    s = default_row.copy()
    target = default_target.copy()
    s["labile_inr"] = True  # L = 1
    scorer = HASBLEDScore()
    output = scorer.calculate(s)
    target["L"] = 1
    target["score"] = 1
    assert all(o == t for o, t in zip(output, target))


def test_hasbled_score_elderly_true(default_row, default_target):
    # test various float type representation
    s = default_row.copy()
    target = default_target.copy()
    scorer = HASBLEDScore()

    s["date_of_birth"] = "31.12.1954"
    s["date_of_discharge"] = "01.01.2020"  # E = 1
    output = scorer.calculate(s)
    target["E"] = 1
    target["score"] = 1
    assert all(o == t for o, t in zip(output, target))

    s["date_of_birth"] = "01.01.1955"  # E = 0
    output = scorer.calculate(s)
    target["E"] = 0
    target["score"] = 0
    assert all(o == t for o, t in zip(output, target))


def test_drugs_true(default_row, default_target):
    # test drugs
    s = default_row.copy()
    target = default_target.copy()
    scorer = HASBLEDScore()

    # D1 = True, D2 = False
    s["med_bleed"] = True  # D1 = 1, D2 = 0
    output = scorer.calculate(s)
    target["D1"] = 1
    target["score"] = 1
    assert all(o == t for o, t in zip(output, target))

    s["alcohol"] = True  # D1 = 1, D2 = 1
    output = scorer.calculate(s)
    target["D2"] = 1
    target["score"] = 2
    assert all(o == t for o, t in zip(output, target))

    s["med_bleed"] = False  # D1 = 0, D2 = 1
    output = scorer.calculate(s)
    target["D1"] = 0
    target["score"] = 1
    assert all(o == t for o, t in zip(output, target))
