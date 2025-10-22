# src/scoring/euroscoreii.py

import numbers
from numpy import exp
from pandas import concat, Series

from src.scoring.base import RiskScore, RiskScoreFactory


@RiskScoreFactory.register("euroscoreii")
class EuroSCOREII(RiskScore):
    """
    EuroSCORE II calculator for estimating in-hospital mortality
    after cardiac surgery.

    Reference:
        Nashef SA, Roques F, Sharples LD, et al.
        EuroSCORE II. European Journal of Cardio-Thoracic Surgery.
        Volume 41, Issue 4, April 2012, Pages 734–745.
        https://doi.org/10.1093/ejcts/ezs043

    Overview:
        EuroSCORE II is a validated logistic regression risk model used to predict
        perioperative mortality among patients undergoing major cardiac surgery.
        It incorporates demographic data, cardiac and non-cardiac comorbidities,
        characteristics of the cardiac procedure, and selected preoperative labs.

    Source of Coefficients:
        Based on Table 6 of the cited publication, which provides adjusted
        regression coefficients from the multivariate logistic regression.

    Input features include (abbreviation in original publication):
        - age (age)
        - Gender (female)
        - Severe chronic pulmonary disease (cpd)
        - Extracardiac arteriopathy (eca)
        - Poor mobility (NM mob)
        - Previous cardiac surgery (redo)
        - Active endocarditis (ae)
        - Critical state (critical)
        - Renal dysfunction (Renal dysfunction)
        - Insulin-dependent diabetes mellitus (iddm)
        - CCS angina class = 4 (ccs4)
        - Left ventricular function (LV function)
        - Recent MI (recent mi)
        - Pulmonary hypertension (PA systolic pressure)
        - NYHA class (nyha)
        - Thoracic aorta (Thoracic aorta)
        - Urgency of the procedure (urgency)
        - Weight and type of surgery (Weight of procedure)
    """

    _COEFF_MAP = {
        "age": 0.0285181,
        "female": {True: 0.2196434, False: 0},
        "cpd": {True: 0.1886564, False: 0},
        "eca": {True: 0.5360268, False: 0},
        "nm_mob": {True: 0.2407181, False: 0},
        "redo": {True: 1.118599, False: 0},
        "ae": {True: 0.6194522, False: 0},
        "critical": {True: 1.086517, False: 0},
        "renal_dysfunction": {
            "no": 0,
            "cc51-85": 0.303553,  # = moderately impaired
            "on dialysis": 0.6421508,
            "cc≤50": 0.8592256,
        },
        "iddm": {True: 0.3542749, False: 0},
        "ccs4": {True: 0.2226147, False: 0},
        "lv_function": {
            "normal": 0.0,
            "moderate": 0.3150652,
            "poor": 0.8084096,
            "very poor": 0.9346919,
        },
        "recent_mi": {True: 0.1528943, False: 0},
        "pa_systolic_pressure": {  # PA systolic pressure
            "normal": 0.0,
            "31–55mmHg": 0.1788899,  # moderate
            "≥55mmHg": 0.3491475,  # severe
        },
        "nyha": {
            1: 0.0,
            2: 0.1070545,
            3: 0.2958358,
            4: 0.5597929,
        },
        "thoracic_aorta": {True: 0.6527205, False: 0},  # 0.6527205
        "urgency": {
            "elective": 0.0,
            "urgent": 0.3174673,
            "emergency": 0.7039121,
            "salvage": 1.362947,
        },
        "weight_of_procedure": {
            "isolated CABG": 0.0,
            "1 non-CABG": 0.0062118,
            "2": 0.5521478,
            "3+": 0.9724533,
        },
        "constant": -5.324537,
    }

    def __init__(self):
        super().__init__(name="EuroSCORE II")

    def calculate(self, llm_output_row: Series) -> Series:
        coeffs = self._COEFF_MAP
        # Age points
        # at least one point is given for patients <= 60 according
        # to the original publication (p. 737)
        dob = llm_output_row.get("date_of_birth")
        dod = llm_output_row.get("date_of_discharge")
        age, _ = self.age_from_dates(dob, dod)
        age_coeff = max(age - 59, 1) * coeffs["age"]

        # Is female
        female = self.safe_bool(llm_output_row.get("is_female"))
        female_coeff = coeffs["female"][female]

        # CPD
        copd = self.safe_bool(llm_output_row.get("copd"))
        bronchodilators = self.safe_bool(llm_output_row.get("bronchodilators"))
        steroids = self.safe_bool(llm_output_row.get("steroids"))
        cpd = copd | bronchodilators | steroids
        cpd_coeff = coeffs["cpd"][cpd]

        # ECA
        claudication = self.safe_bool(llm_output_row.get("claudication"))
        carotid = self.safe_bool(llm_output_row.get("carotid"))
        procedure = self.safe_bool(llm_output_row.get("procedure"))
        eca = claudication | carotid | procedure
        eca_coeff = coeffs["eca"][eca]

        # NM_mob: poor mobility
        nm_mob = self.safe_bool(llm_output_row.get("poor_mobility"))
        nm_mob_coeff = coeffs["nm_mob"][nm_mob]

        # previous cardiac surgery
        redo = self.safe_bool(llm_output_row.get("prev_cardiac_surgery"))
        redo_coeff = coeffs["redo"][redo]

        # Active endocarditis
        ae = self.safe_bool(llm_output_row.get("active_endocarditis"))
        ae_coeff = coeffs["ae"][ae]

        # Critical preoperative state
        critical_rhythm = self.safe_bool(llm_output_row.get("critical_rhythm"))
        critical_cpr = self.safe_bool(llm_output_row.get("critical_cpr"))
        critical_rescue = self.safe_bool(llm_output_row.get("critical_rescue"))
        critical_renal = self.safe_bool(llm_output_row.get("critical_renal"))
        critical = critical_rhythm | critical_cpr | critical_rescue | critical_renal
        critical_coeff = coeffs["critical"][critical]

        # Renal dysfunction
        # if patient is on dialysis, clearance rate is disregarded
        dialysis = self.safe_bool(llm_output_row.get("dialysis"))
        if dialysis:
            renal_dysfunction = "on dialysis"
        else:
            egfr = self.safe_float(llm_output_row.get("egfr"), default=100)
            if egfr <= 50:
                renal_dysfunction = "cc≤50"
            elif egfr <= 85:
                renal_dysfunction = "cc51-85"
            else:
                renal_dysfunction = "no"
        renal_dysfunction_coeff = coeffs["renal_dysfunction"][renal_dysfunction]

        # IDDM
        iddm = self.safe_bool(llm_output_row.get("diabetes_on_insulin"))
        iddm_coeff = coeffs["iddm"][iddm]

        # CCS4
        ccs4 = self.safe_bool(llm_output_row.get("ccs4"))
        ccs4_coeff = coeffs["ccs4"][ccs4]

        # LV function
        lvef = self.safe_float(llm_output_row.get("lvef", default=-1))
        lvef_literal = llm_output_row.get("lvef_literal", "missing")
        lv_function = "normal"
        if lvef_literal != "missing" and lvef_literal in coeffs["lv_function"].keys():
            lv_function = lvef_literal
        elif lvef > -1:
            if lvef >= 51:
                lv_function = "normal"
            elif lvef >= 31:
                lv_function = "moderate"
            elif lvef >= 21:
                lv_function = "poor"
            elif lvef > 0.0:
                lv_function = "very poor"
        lv_function_coeff = coeffs["lv_function"][lv_function]

        # Recent MI
        recent_mi = self.safe_bool(llm_output_row.get("recent_mi"))
        recent_mi_coeff = coeffs["recent_mi"][recent_mi]

        # PA systolic pressure
        spap = self.safe_float(llm_output_row.get("spap"), default=-1)
        spap_echo = self.safe_float(
            llm_output_row.get("spap_echo"), default="normal", missing_value="normal"
        )
        if spap == -1 and isinstance(spap_echo, numbers.Number):
            assert (
                spap_echo >= 10 and spap_echo <= 100
            ), "spap_echo must be between 10 and 100"
            # account for underestimation in echo measurements
            spap = spap_echo + 8
        pa_systolic_pressure = "normal"
        if spap > -1:
            if spap >= 55:
                pa_systolic_pressure = "≥55mmHg"
            elif spap >= 31:
                pa_systolic_pressure = "31–55mmHg"
        pa_systolic_pressure_coeff = coeffs["pa_systolic_pressure"][
            pa_systolic_pressure
        ]

        # NYHA class
        nyha = self.safe_nyha(llm_output_row.get("nyha", 1))
        nyha_coeff = coeffs["nyha"][nyha]

        # Thoracic aorta
        thoracic_aorta = self.safe_bool(llm_output_row.get("thoracic_surgery"))
        thoracic_aorta_mm = self.safe_float(
            llm_output_row.get("thoracic_aorta_mm"),
            default="missing",
            missing_value="missing",
        )
        if not thoracic_aorta and thoracic_aorta_mm != "missing":
            if thoracic_aorta_mm > 45:
                thoracic_aorta = True
        thoracic_aorta_coeff = coeffs["thoracic_aorta"][thoracic_aorta]
        print("thorace_aorta_coeff", thoracic_aorta_coeff)

        # Urgency of operation
        urgent = self.safe_bool(llm_output_row.get("urgency_urgent"))
        emergency = self.safe_bool(llm_output_row.get("urgency_emergency"))
        salvage = self.safe_bool(llm_output_row.get("urgency_salvage"))
        urgency = "urgent" if urgent else "elective"
        urgency = "emergency" if emergency else urgency
        urgency = "salvage" if salvage else urgency
        urgency_coeff = coeffs["urgency"][urgency]

        # Weight of procedure
        # assume patient is scheduled for isolated CABG if no procedure is
        # explicitly mentioned in the report
        weight_cabg = self.safe_bool(llm_output_row.get("weight_cabg"))
        weight_valve = self.safe_bool(llm_output_row.get("weight_valve"))
        weight_aorta = self.safe_bool(llm_output_row.get("weight_aorta"))
        weight_maze = self.safe_bool(llm_output_row.get("weight_maze"))
        weight_defect = self.safe_bool(llm_output_row.get("weight_defect"))
        weight_tumor = self.safe_bool(llm_output_row.get("weight_tumor"))
        procedures_non_cabg = sum(
            [
                weight_valve,
                weight_aorta,
                weight_maze,
                weight_defect,
                weight_tumor,
            ]
        )
        if procedures_non_cabg == 0:
            weight_of_procedure = "isolated CABG"
        else:  # procedures_non_cabg > 0!
            if procedures_non_cabg == 1 and weight_cabg == 0:
                weight_of_procedure = "1 non-CABG"
            elif procedures_non_cabg + weight_cabg == 2:
                weight_of_procedure = "2"
            else:
                weight_of_procedure = "3+"
        
        weight_of_procedure_coeff = coeffs["weight_of_procedure"][weight_of_procedure]

        items = {
            "age": age,
            "is_female": female,
            "cpd": cpd,
            "eca": eca,
            "nm_mob": nm_mob,
            "redo": redo,
            "ae": ae,
            "critical": critical,
            "renal_dysfunction": renal_dysfunction,
            "iddm": iddm,
            "ccs4": ccs4,
            "lv_function": lv_function,
            "recent_mi": recent_mi,
            "pa_systolic_pressure": pa_systolic_pressure,
            "nyha": nyha,
            "thoracic_aorta": thoracic_aorta,
            "urgency": urgency,
            "weight_of_procedure": weight_of_procedure,
        }
        df_items = Series(items)
        coeffs = {
            "age_coeff": age_coeff,
            "is_female_coeff": female_coeff,
            "cpd_coeff": cpd_coeff,
            "eca_coeff": eca_coeff,
            "nm_mob_coeff": nm_mob_coeff,
            "redo_coeff": redo_coeff,
            "ae_coeff": ae_coeff,
            "critical_coeff": critical_coeff,
            "renal_dysfunction_coeff": renal_dysfunction_coeff,
            "iddm_coeff": iddm_coeff,
            "ccs4_coeff": ccs4_coeff,
            "lv_function_coeff": lv_function_coeff,
            "recent_mi_coeff": recent_mi_coeff,
            "pa_systolic_pressure_coeff": pa_systolic_pressure_coeff,
            "nyha_coeff": nyha_coeff,
            "thoracic_aorta_coeff": thoracic_aorta_coeff,
            "urgency_coeff": urgency_coeff,
            "weight_of_procedure_coeff": weight_of_procedure_coeff,
            "constant": coeffs["constant"],
        }
        df_coeffs = Series(coeffs)
        coeffs_sum = df_coeffs.sum()
        print("coeffs:", {d: v for d, v in coeffs.items() if v != 0})
        print("coeffs_sum:", coeffs_sum)
        df_coeffs["score"] = exp(coeffs_sum) / (1.0 + exp(coeffs_sum)) * 100
        return concat([df_items, df_coeffs])
