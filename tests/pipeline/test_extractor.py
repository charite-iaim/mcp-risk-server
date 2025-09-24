# /tests/pipeline/test_extractor.py


import pytest

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


@pytest.fixture
def cha2ds2vasc_completions():
    return {
        "date_of_birth": ('{"date_of_birth": "01.02.1954"}', "01.02.1954"),
        "date_of_discharge": ('{"date_of_discharge": "28.09.2025"}', "28.09.2025"),
        "is_female": ('{"is_female": 1}', 1),
        "chf_symptoms": ('{"chf_symptoms": 1}', 1),
        "chf_history": ('{"chf_history": 1}', 1),
        "chf_lvef": ('{"chf_lvef": 0}', 0),
        "hypertension_diagnosis": ('{"hypertension_diagnosis": 1}', 1),
        "hypertension_medication": ('{"hypertension_medication": 1}', 1),
        "stroke_history": ('{"stroke_history": 1}', 1),
        "diabetes": ('{"diabetes": 1}', 1),
        "hba1c": ('{"hba1c": 6.9}', 6.9),
        "vascular_disease": ('{"vascular_disease": 1}', 1),
    }


@pytest.fixture
def euroscoreii_completions():
    return {
        "date_of_birth": ('{"date_of_birth": "15.07.1948"}', "15.07.1948"),
        "date_of_discharge": ('{"date_of_discharge": "20.09.2025"}', "20.09.2025"),
        "is_female": ('{"is_female": 0}', 0),
        "bronchodilators": ('{"bronchodilators": 1}', 1),
        "steroids": ('{"steroids": 1}', 1),
        "claudication": ('{"claudication": 0}', 0),
        "carotid": ('{"carotid": 0}', 0),
        "procedure": ('{"procedure": 1}', 1),
        "poormobility": ('{"poormobility": 0}', 0),
        "prev_cardiac_surgery": ('{"prev_cardiac_surgery": 1}', 1),
        "active_endocarditis": ('{"active_endocarditis": 0}', 0),
        "critical_rhythm": ('{"critical_rhythm": 0}', 0),
        "critical_cpr": ('{"critical_cpr": 0}', 0),
        "critical_rescue": ('{"critical_rescue": 0}', 0),
        "critical_renal": ('{"critical_renal": 1}', 1),
        "dialysis": ('{"dialysis": 0}', 0),
        "egfr": ('{"egfr": 58}', 58),
        "diabetes_on_insulin": ('{"diabetes_on_insulin": 1}', 1),
        "ccs4": ('{"ccs4": 0}', 0),
        "lvef": ('{"lvef": 45}', 45),
        "lvefliteral": ('{"lvefliteral": "moderate"}', "moderate"),
        "recent_mi": ('{"recent_mi": 1, "recent_mi": 1}', 1),
        "date_of_recent_mi": ('{"date_of_recent_mi": "01.08.2025"}', "01.08.2025"),
        "spap": ('{"spap": 34}', 34),
        "spap_echo": ('{"spap_echo": 36}', 36),
        "nyha": ('{"nyha": "II"}', "II"),
        "thoracic_surgery": ('{"thoracic_surgery": 0}', 0),
        "thoracic_aorta_mm": ('{"thoracic_aorta_mm": 42}', 42),
        "urgency_elective": ('{"urgency_elective": 1}', 1),
        "urgency_urgent": ('{"urgency_urgent": 0}', 0),
        "urgency_emergency": ('{"urgency_emergency": 0}', 0),
        "urgency_salvage": ('{"urgency_salvage": 0}', 0),
        "weight_cabg": ('{"weight_cabg": 0}', 0),
        "weight_valve": ('{"weight_valve": 1}', 1),
        "weight_aorta": ('{"weight_aorta": 0}', 0),
        "weight_maze": ('{"weight_maze": 0}', 0),
        "weight_defect": ('{"weight_defect": 0}', 0),
        "weight_tumor": ('{"weight_tumor": 0}', 0),
    }


@pytest.fixture
def all_score_completions(
    hasbled_completions, cha2ds2vasc_completions, euroscoreii_completions
):
    return {
        "hasbled": hasbled_completions,
        "cha2ds2vasc": cha2ds2vasc_completions,
        "euroscoreii": euroscoreii_completions,
    }


@pytest.mark.parametrize("score_key", ["hasbled", "cha2ds2vasc", "euroscoreii"])
def test_extraction(all_score_completions, score_key):
    completions = all_score_completions[score_key]

    extractor = Extractor()
    for item, values in completions.items():
        response, value_extracted = values
        value_dict = extractor(response)
        assert len(value_dict) == 1
        assert item in value_dict
        assert value_dict.get(item) == value_extracted
