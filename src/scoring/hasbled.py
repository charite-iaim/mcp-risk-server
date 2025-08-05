# mcp/scoring/hasbled.py

from pandas import Series

from src.scoring.base import RiskScore, RiskScoreFactory


@RiskScoreFactory.register("hasbled")
class HASBLEDScore(RiskScore):
    """
    HAS-BLED score calculator for estimating 1-year risk of major bleeding
    in patients with atrial fibrillation.

    Reference:
        Pisters R, Lane DA, Nieuwlaat R, De Vos CB, Crijns HJGM, Lip GYH.
        A novel user-friendly score (HAS-BLED) to assess 1-year risk of major
        bleeding in patients with atrial fibrillation: the Euro Heart Survey.
        Chest. 2010;138(5):1093–1100.
        https://doi.org/10.1378/chest.10-0134

    Overview:
        The clinical risk scoring table based on validated multivariate
        analysis of bleeding risk factors listed below.

    Features typically include (abbreviation from original publication):
        - Hypertension (H)
        - Abnormal liver (A1) or renal (A2) function
        - History of stroke (S)
        - Bleeding tendency (B)
        - Labile INR (L)
        - Age > 65 (E)
        - Use of alcohol (D1) or drugs (D2)
    """

    def __init__(self):
        super().__init__(name="HAS-BLED")

    def calculate(self, llm_output_row: Series) -> Series:

        # H: Hypertension (systolic > 160 mmHg)
        sys_bp = self.safe_float(llm_output_row.get("sys_bp"))
        H = 1.0 if sys_bp > 160 else 0.0

        # A1: Abnormal renal function
        dialysis = self.safe_bool(llm_output_row.get("dialysis"))
        renal_disease = self.safe_bool(llm_output_row.get("renal_disease"))
        creatinine = self.safe_float(llm_output_row.get("creatinine"))
        unit_mg_per_dL = self.safe_bool(llm_output_row.get("unit_mg_per_dL"))
        creatinine_exceeded1 = (creatinine >= 200) and (not unit_mg_per_dL)
        creatinine_exceeded2 = (creatinine >= 2.26) and unit_mg_per_dL
        creatinine_exceeded = creatinine_exceeded1 | creatinine_exceeded2
        A1 = dialysis | renal_disease | creatinine_exceeded

        # A2: Abnormal liver function
        # if no ULN (AST, ALT, ALP) is available, use only bilirubin,
        # otherwise if at least one is available and is -- in combination with
        # excessive bilirubin levels -- exceeding upper limit of normal range
        # at least three times then a point is given
        liver_disease = self.safe_bool(llm_output_row.get("liver_disease"))

        bilirubin = self.safe_float(llm_output_row.get("bilirubin"), default=0.0)
        bilirubin_exceeded = bilirubin > 2.6

        # upper limit of normal range for AST according to
        # emedicine.medscape.com/article/2087224-overview
        ast = self.safe_float(llm_output_row.get("ast"), default=0.0)
        ast_exceeded = ast >= 3 * 35

        # use upper limit for normal range for ALT according to
        # www.labcorp.com/assets/5286
        alt = self.safe_float(llm_output_row.get("alt"), default=0.0)
        is_female = self.safe_bool(llm_output_row.get("is_female"))
        alt_exceeded = alt >= 3 * (33 if is_female else 45)

        # use upper limits for normal range for ALP according to EU/UK != US
        # www.uhnm.nhs.uk/our-services/pathology/tests/alkaline-phosphatase
        alp = self.safe_float(llm_output_row.get("alp"), default=0.0)
        alp_exceeded = alp >= 3 * 130

        uln_exceeded = ast_exceeded | alt_exceeded | alp_exceeded
        no_uln = sum([ast, alt, alp]) == 0
        bilirubin_only = no_uln & bilirubin_exceeded
        bilirubin_and_uln = (not no_uln) & bilirubin_exceeded & uln_exceeded
        A2 = liver_disease | bilirubin_only | bilirubin_and_uln

        # S: Stroke
        S = self.safe_bool(llm_output_row.get("stroke_history"))

        # B: Bleeding history + low hemoglobin
        bleeding_history = self.safe_bool(llm_output_row.get("bleeding_history"))
        hemoglobin_raw = llm_output_row.get("hemoglobin", 100)
        hemoglobin = self.safe_float(hemoglobin_raw, default=100)
        threshold_female = is_female & (hemoglobin < 12)
        threshold_male = (not is_female) & (hemoglobin < 13)
        bleeding_risk = threshold_female | threshold_male
        B = bleeding_history | bleeding_risk

        # L: Labile INR
        L = self.safe_bool(llm_output_row.get("labile_inr"))

        # E: Elderly (> 65 years)
        dob = llm_output_row.get("date_of_birth")
        dod = llm_output_row.get("date_of_discharge")
        age_int, delta_days = self.age_from_dates(dob, dod)
        E = (age_int > 65) | ((age_int == 65) & (delta_days > 0))

        # D: Drugs or Alcohol
        D1 = self.safe_bool(llm_output_row.get("med_bleed"))
        D2 = self.safe_bool(llm_output_row.get("alcohol"))

        # Pack items and calculated values into Series
        items = {
            "H": H,
            "A1": A1,
            "A2": A2,
            "S": S,
            "B": B,
            "L": L,
            "E": E,
            "D1": D1,
            "D2": D2,
        }

        df_score = Series(items)
        df_score["score"] = df_score.sum()
        return df_score
