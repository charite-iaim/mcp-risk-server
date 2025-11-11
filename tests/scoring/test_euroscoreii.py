# tests/scoring/test_euroscoreii.py
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import yaml

from src.scoring.euroscoreii import EuroSCOREII


@pytest.fixture
def llm_row():
    return pd.Series(
        {
            "date_of_birth": "01.01.1970",
            "date_of_discharge": "01.01.2020",
            "is_female": 0,
            "copd": 0,
            "bronchodilators": 0,
            "steroids": 0,
            "claudication": 0,
            "carotid": 0,
            "procedure": 0,
            "poor_mobility": 0,
            "prev_cardiac_surgery": 0,
            "active_endocarditis": 0,
            "critical_rhythm": 0,
            "critical_cpr": 0,
            "critical_rescue": 0,
            "critical_renal": 0,
            "dialysis": 0,
            "egfr": 88,
            "diabetes_on_insulin": 0,
            "ccs4": 0,
            "lvef": 60,
            "lvef_literal": "missing",
            "recent_mi": 0,
            "spap": 30,
            "spap_echo": 22,
            "nyha": "I",
            "thoracic_surgery": 0,
            "thoracic_aorta_mm": 44,
            "urgency_elective": 1,
            "urgency_urgent": 0,
            "urgency_emergency": 0,
            "urgency_salvage": 0,
            "weight_cabg": 0,
            "weight_valve": 0,
            "weight_aorta": 0,
            "weight_maze": 0,
            "weight_defect": 0,
            "weight_tumor": 0,
        }
    )


@pytest.fixture
def calc_row():
    return pd.Series(
        {
            "age": 50,
            "is_female": 0,
            "cpd": 0,
            "eca": 0,
            "nm_mob": 0,
            "redo": 0,
            "ae": 0,
            "critical": 0,
            "renal_dysfunction": "no",
            "iddm": 0,
            "ccs4": 0,
            "lv_function": "normal",
            "recent_mi": 0,
            "pa_systolic_pressure": "normal",
            "nyha": 1,
            "thoracic_aorta": 0,
            "urgency": "elective",
            "weight_of_procedure": "isolated CABG",
            "score": 0.5,
        }
    )


def test_euroscoreii_template_keys(llm_row):
    template_file = Path("src") / "prompts" / "euroscoreii_template.yaml"
    with open(template_file) as f:
        template = yaml.safe_load(f)
    template_keys = set(template.keys())
    template_keys.remove("intro")
    llm_keys = set(llm_row.index)
    assert template_keys == llm_keys


def test_euroscoreii_default(llm_row, calc_row):
    scorer = EuroSCOREII()
    output = scorer.calculate(llm_row)
    assert len(set(calc_row.keys()).difference(set(output.keys()))) == 0
    for k in calc_row.keys():
        if k == "score":
            assert np.isclose(output[k], calc_row[k], atol=0.01)
        else:
            assert output[k] == calc_row[k]


def test_euroscoreii_age(llm_row, calc_row):
    # test age factor
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()

    src["date_of_birth"] = "01.01.1960"  # age 60
    src["date_of_discharge"] = "01.01.2020"
    tgt["age"] = 60
    tgt["score"] = 0.5
    output = scorer.calculate(src)
    assert output["age"] == tgt["age"]
    assert np.isclose(output["score"], tgt["score"], atol=0.005)

    src["date_of_birth"] = "01.01.1959"  # age 61
    src["date_of_discharge"] = "01.01.2020"
    tgt["age"] = 61
    tgt["score"] = 0.51
    output = scorer.calculate(src)
    assert output["age"] == tgt["age"]
    assert np.isclose(output["score"], tgt["score"], atol=0.005)

    src["date_of_birth"] = "01.01.1940"  # age 80
    src["date_of_discharge"] = "01.01.2020"
    tgt["age"] = 80
    tgt["score"] = 0.88
    output = scorer.calculate(src)
    assert output["age"] == tgt["age"]
    assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_is_female(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    src["is_female"] = 1
    tgt["is_female"] = 1
    tgt["score"] = 0.62
    output = scorer.calculate(src)
    assert output["is_female"] == tgt["is_female"]
    assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_cpd(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    for v1, v2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        src["bronchodilators"] = v1
        src["steroids"] = v2
        tgt["cpd"] = 1 if (v1 or v2) else 0
        tgt["score"] = 0.6 if tgt["cpd"] else 0.5
        output = scorer.calculate(src)
        assert output["cpd"] == tgt["cpd"]
        assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_eca(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    for c in itertools.product([0, 1], repeat=3):
        v1, v2, v3 = c
        src["claudication"] = v1
        src["carotid"] = v2
        src["procedure"] = v3
        tgt["eca"] = 1 if (v1 or v2 or v3) else 0
        tgt["score"] = 0.85 if tgt["eca"] else 0.5
        output = scorer.calculate(src)
        assert output["eca"] == tgt["eca"]
        assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_nm_mob(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    src["poor_mobility"] = True
    tgt["nm_mob"] = 1
    tgt["score"] = 0.63
    output = scorer.calculate(src)
    assert output["nm_mob"] == tgt["nm_mob"]
    assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_redo(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    src["prev_cardiac_surgery"] = 1
    tgt["redo"] = 1
    tgt["score"] = 1.51
    output = scorer.calculate(src)
    assert output["redo"] == tgt["redo"]
    assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_ae(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    src["active_endocarditis"] = 1
    tgt["ae"] = 1
    tgt["score"] = 0.92
    output = scorer.calculate(src)
    assert output["ae"] == tgt["ae"]
    assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_critical(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    for c in itertools.product([0, 1], repeat=4):
        v1, v2, v3, v4 = c
        src["critical_rhythm"] = v1
        src["critical_cpr"] = v2
        src["critical_rescue"] = v3
        src["critical_renal"] = v4
        critical = 1 if (v1 or v2 or v3 or v4) else 0
        tgt["critical"] = critical
        tgt["score"] = 1.46 if critical else 0.5
        output = scorer.calculate(src)
        assert output["critical"] == tgt["critical"]
        assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_renal_dysfunction(llm_row, calc_row):
    # TODO: continue here
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()

    for label, egfr, score in [
        ("no", 88, 0.5),
        ("cc51-85", 55, 0.67),  # moderate
        ("cc≤50", 45, 1.17),  # severe
        ("on dialysis", 49, 0.94),
    ]:
        src["egfr"] = egfr
        src["dialysis"] = 1 if label == "on dialysis" else 0
        tgt["renal_dysfunction"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert output["renal_dysfunction"] == tgt["renal_dysfunction"]
        assert np.isclose(output["score"], tgt["score"], atol=0.005)
    output = scorer.calculate(src)


def test_euroscoreii_iddm(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    src["diabetes_on_insulin"] = 1
    tgt["iddm"] = 1
    tgt["score"] = 0.71
    output = scorer.calculate(src)
    assert output["iddm"] == tgt["iddm"]
    assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_ccs4(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    src["ccs4"] = 1
    tgt["ccs4"] = 1
    tgt["score"] = 0.62
    output = scorer.calculate(src)
    assert output["ccs4"] == tgt["ccs4"]
    assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_lv_function(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    for label, lvef, score in [
        ("normal", 51, 0.5),  # = good
        ("moderate", 32, 0.68),
        ("poor", 29, 1.11),
        ("very poor", 20.9, 1.26),
    ]:
        src["lvef"] = lvef
        src["lvef_literal"] = "missing"
        tgt["lv_function"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert output["lv_function"] == tgt["lv_function"]
        assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_recent_mi(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    src["recent_mi"] = 1
    tgt["recent_mi"] = 1
    tgt["score"] = 0.58
    output = scorer.calculate(src)
    assert output["recent_mi"] == tgt["recent_mi"]
    assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_pa_systolic_pressure(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    for label, spap, spap_echo, score in [
        ("normal", "missing", "normal", 0.5),
        ("normal", "missing", 22, 0.5),
        ("normal", 30, 22, 0.5),
        ("31–55mmHg", "missing", 36, 0.6),
        ("31–55mmHg", 45, 36, 0.6),
        ("≥55mmHg", "missing", 51, 0.71),
        ("≥55mmHg", 55, 40, 0.71),
    ]:
        src["spap"] = spap
        src["spap_echo"] = spap_echo
        tgt["pa_systolic_pressure"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert output["pa_systolic_pressure"] == tgt["pa_systolic_pressure"]
        assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_nyha(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    for label, nyha, score in [
        (1, "I", 0.5),
        (2, "II", 0.55),
        (3, "III", 0.67),
        (4, "IV", 0.87),
    ]:
        src["nyha"] = nyha
        tgt["nyha"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert output["nyha"] == tgt["nyha"]
        assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_thoracic_aorta(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    for label, thoracic_surgery, thoracic_aorta_mm, score in [
        (False, 0, "missing", 0.5),
        (False, 0, 45, 0.5),
        (True, 0, 46, 0.9534),
        (True, 1, 45, 0.9534),
    ]:
        src["thoracic_surgery"] = thoracic_surgery
        src["thoracic_aorta_mm"] = thoracic_aorta_mm
        tgt["thoracic_aorta"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert output["thoracic_aorta"] == tgt["thoracic_aorta"]
        assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_urgency(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    for (
        label,
        urgency_elective,
        urgency_urgent,
        urgency_emergency,
        urgency_salvage,
        score,
    ) in [
        ("elective", 1, 0, 0, 0, 0.5),
        ("elective", 0, 0, 0, 0, 0.5),
        ("urgent", 0, 1, 0, 0, 0.6837),
        ("urgent", 1, 1, 0, 0, 0.6837),
        ("emergency", 0, 0, 1, 0, 1.0),
        ("emergency", 1, 1, 1, 0, 1.0),
        ("salvage", 0, 0, 0, 1, 1.92),
        ("salvage", 0, 0, 1, 1, 1.92),
        ("salvage", 0, 1, 0, 1, 1.92),
        ("salvage", 1, 0, 0, 1, 1.92),
    ]:
        src["urgency_elective"] = urgency_elective
        src["urgency_urgent"] = urgency_urgent
        src["urgency_emergency"] = urgency_emergency
        src["urgency_salvage"] = urgency_salvage
        tgt["urgency"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert output["urgency"] == tgt["urgency"]
        assert np.isclose(output["score"], tgt["score"], atol=0.005)


def test_euroscoreii_weight_of_procedure(llm_row, calc_row):
    src = llm_row.copy()
    tgt = calc_row.copy()
    scorer = EuroSCOREII()
    for (
        label,
        weight_cabg,
        weight_valve,
        weight_aorta,
        weight_maze,
        weight_defect,
        weight_tumor,
        score,
    ) in [
        ("isolated CABG", 0, 0, 0, 0, 0, 0, 0.5),
        ("1 non-CABG", 0, 1, 0, 0, 0, 0, 0.5),
        ("1 non-CABG", 0, 0, 1, 0, 0, 0, 0.5),
        ("1 non-CABG", 0, 0, 0, 1, 0, 0, 0.5),
        ("1 non-CABG", 0, 0, 0, 0, 1, 0, 0.5),
        ("1 non-CABG", 0, 0, 0, 0, 0, 1, 0.5),
        ("2", 1, 1, 0, 0, 0, 0, 0.86),
        ("2", 0, 1, 0, 1, 0, 0, 0.86),
        ("2", 0, 0, 1, 0, 0, 1, 0.86),
        ("2", 0, 0, 0, 0, 1, 1, 0.86),
        ("3+", 0, 1, 1, 0, 1, 0, 1.31),
        ("3+", 0, 1, 1, 0, 1, 1, 1.31),
        ("3+", 0, 0, 1, 1, 1, 1, 1.31),
        ("3+", 0, 1, 1, 1, 1, 1, 1.31),
    ]:
        src["weight_cabg"] = weight_cabg
        src["weight_valve"] = weight_valve
        src["weight_aorta"] = weight_aorta
        src["weight_maze"] = weight_maze
        src["weight_defect"] = weight_defect
        src["weight_tumor"] = weight_tumor
        tgt["weight_of_procedure"] = label
        tgt["score"] = score
        output = scorer.calculate(src)
        assert output["weight_of_procedure"] == tgt["weight_of_procedure"]
        assert np.isclose(output["score"], tgt["score"], atol=0.005)
