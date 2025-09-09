# src/scoring/cha2ds2vasc.py
from pandas import Series

from src.scoring.base import RiskScore, RiskScoreFactory


@RiskScoreFactory.register("cha2ds2vasc")
class CHA2DS2VAScScore(RiskScore):
    """
    CHA₂DS₂-VASc score calculator for estimating the risk of stroke and
    thromboembolism in patients with atrial fibrillation.

    Reference:
        Lip GYH, Nieuwlaat R, Pisters R, Lane DA, Crijns HJGM.
        Refining clinical risk stratification for predicting stroke and
        thromboembolism in atrial fibrillation using a novel risk factor-based
        approach: the Euro Heart Survey on atrial fibrillation.
        Chest. 2010;137(2):263–272.
        https://doi.org/10.1378/chest.09-1584

    Overview:
        The CHA₂DS₂-VASc score is a validated clinical risk model improving
        on the original CHADS₂ system by incorporating additional stroke
        risk factors. It supports evidence-based decisions on anticoagulation
        in atrial fibrillation patients for primary and secondary prevention.

    Model Inputs (abbreviation from original publication):
        - Congestive heart failure (C)
        - Hypertension (H)
        - Age ≥75 years (A2)
        - Diabetes mellitus (D)
        - Prior stroke, TIA, or thromboembolism (S2)
        - Vascular disease like MI, PAD, or aortic plaque (V)
        - Age 65–74 years (A)
        - Sex category, is female (Sc)
    """

    def __init__(self):
        super().__init__(name="CHA₂DS₂-VASc")

    def calculate(self, llm_output_row: Series) -> Series:
        # C: Congestive heart failure
        chf_symptoms = self.safe_bool(llm_output_row.get("chf_symptoms"))
        chf_history = self.safe_bool(llm_output_row.get("chf_history"))
        chf_lvef = self.safe_bool(llm_output_row.get("chf_lvef"))
        C = chf_symptoms | chf_history | chf_lvef

        # H: hypertension history
        hypertension_diagnosis = self.safe_bool(
            llm_output_row.get("hypertension_diagnosis")
        )
        hypertension_medication = self.safe_bool(
            llm_output_row.get("hypertension_medication")
        )
        H = hypertension_diagnosis | hypertension_medication

        # A2: Patient age >= 75
        dob = llm_output_row.get("date_of_birth")
        dod = llm_output_row.get("date_of_discharge")
        age, _ = self.age_from_dates(dob, dod)
        A2 = 2 if age >= 75 else 0

        # D: Diabetes
        diabetes = self.safe_bool(llm_output_row.get("diabetes"))
        hba1c = self.safe_float(llm_output_row.get("hba1c"))
        D = diabetes | (hba1c >= 6.5)

        # S2: Stroke/TIA/Thromboembolism
        S2 = self.safe_bool(llm_output_row.get("stroke_history"))
        S2 = 2 if S2 else 0

        # V: Vascular disease
        V = self.safe_bool(llm_output_row.get("vascular_disease"))

        # A: Age in [65:74]
        A = 1 if ((age < 75) and (age > 64)) else 0

        # Sc: Sex category
        Sc = self.safe_bool(llm_output_row.get("is_female"))

        # Pack items and calculated values into Series
        items = {"C": C, "H": H, "A2": A2, "D": D, "S2": S2, "V": V, "A": A, "Sc": Sc}

        df_score = Series(items)
        df_score["score"] = df_score.sum()
        return df_score
