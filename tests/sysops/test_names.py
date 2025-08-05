# mcp/tests/sysops/test_names.py

import pytest

from src.sysops import names


@pytest.mark.parametrize(
    "input_name,expected",
    [
        ("HAS-BLED", "hasbled"),
        ("CHA2DS2-VASc", "cha2ds2vasc"),
        ("EuroSCORE II", "euroscoreii"),
        ("has-bled", "hasbled"),
        ("cha2ds2-vasc", "cha2ds2vasc"),
        ("euroscore ii", "euroscoreii"),
        ("hasbled", "hasbled"),
        ("cha2ds2vasc", "cha2ds2vasc"),
        ("euroscoreii", "euroscoreii"),
        ("Some Score", "somescore"),
        ("foo-bar", "foobar"),
        ("  HAS-BLED  ", "hasbled"),  # whitespace test
    ],
)
def test_get_score_str(input_name, expected):
    assert names.get_score_str(input_name.strip()) == expected


@pytest.mark.parametrize(
    "score_str,expected",
    [
        ("hasbled", "HAS-BLED"),
        ("cha2ds2vasc", "CHA₂DS₂-VASc"),
        ("euroscoreii", "EuroSCORE II"),
        ("foo", "foo"),
    ],
)
def test_get_str_representation(score_str, expected):
    assert names.get_str_representation(score_str) == expected
