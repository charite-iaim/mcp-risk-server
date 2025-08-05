# src/sysops/names.py

# display to string mapping
SCORE_STR = {
    "HAS-BLED": "hasbled",
    "CHA2DS2-VASc": "cha2ds2vasc",
    "EuroSCORE II": "euroscoreii",
    "has-bled": "hasbled",
    "cha2ds2-vasc": "cha2ds2vasc",
    "euroscore ii": "euroscoreii",
    "hasbled": "hasbled",
    "cha2ds2vasc": "cha2ds2vasc",
    "euroscoreii": "euroscoreii",
}

# string to display mapping
SCORE_DISPLAY = {
    "hasbled": "HAS-BLED",
    "cha2ds2vasc": "CHA₂DS₂-VASc",
    "euroscoreii": "EuroSCORE II",
}


def get_score_str(score_name: str) -> str:
    """
    Get the score string representation for a given score name.
    """
    score = score_name.lower()
    if score in SCORE_STR:
        return SCORE_STR[score]
    else:
        return score.replace(" ", "").replace("-", "").lower()


def get_str_representation(score_str: str) -> str:
    """
    Get the display representation of the risk score.
    """
    return SCORE_DISPLAY.get(score_str, score_str)
