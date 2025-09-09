# mcp/scoring/base.py

from abc import ABC, abstractmethod
from datetime import datetime
import logging
from pandas import isna, Series


class RiskScoreFactory:
    _registry = {}

    @classmethod
    def register(cls, key):
        def decorator(subclass):
            cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, key, *args, **kwargs):
        if key not in cls._registry:
            raise ValueError(f"Unknown score type: {key}")
        return cls._registry[key](*args, **kwargs)


class RiskScore(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def calculate(self, row: Series) -> Series:
        """Compute the risk score for a given patient row."""
        pass

    def safe_float(self, value, default=0.0, missing_value="missing"):
        if isna(value) or value == missing_value:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            self.logger.warning(
                f"""
                    [safe_float] Unable to interpret value: '{value}'
                      → returning None
                """
            )
        return None

    def safe_int(self, value, default=0, missing_value="missing"):
        if isna(value) or value == missing_value:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            self.logger.warning(
                f"""
                    [safe_int] Unable to interpret value: '{value}'
                      → returning None
                """
            )
        return None
    
    def safe_nyha(self, value, default=1, missing_value="missing"):
        if isna(value) or value == missing_value:
            return default
        if value in {"I", "II", "III", "IV"}:
            match value:
                case "I":
                    return 1
                case "II":
                    return 2
                case "III":
                    return 3
                case "IV":
                    return 4
        try:
            return int(value)
        except (TypeError, ValueError):
            self.logger.warning(
                f"""
                    [safe_nyha] Unable to interpret value: '{value}'
                      → returning None
                """
            )
        return None

    def safe_bool(self, value, default=False, missing_value="missing"):
        value_str = str(value).strip().lower()
        if isna(value) or value_str == missing_value:
            return default
        if value_str in {"yes", "true", "y"}:
            return True
        elif value_str in {"no", "false", "n"}:
            return False
        else:
            try:
                return bool(float(value))
            except (TypeError, ValueError):
                pass
        self.logger.warning(
            f"""
            [safe_bool] Unable to interpret value: '{value}'
              → returning None
            """
        )
        return None

    def age_from_dates(self, date_of_birth: str, date_of_discharge: str):
        # Calculate age in years and leftover days from date strings
        date_format = "%d.%m.%Y"
        try:
            date1 = datetime.strptime(date_of_birth, date_format)
            date2 = datetime.strptime(date_of_discharge, date_format)
        except (ValueError, TypeError):
            self.logger.error(
                f"""
                [age_from_dates] Unable to interpret date_of_birth 
                ({date_of_birth}) or date_of_discharge ({date_of_discharge}).
                  Expected format is dd.mm.YYYY.
                """
            )
            return False

        # Calculate age in years
        incomplete_year = (date2.month, date2.day) < (date1.month, date1.day)
        age_in_years = date2.year - date1.year - incomplete_year

        # Calculate leftover days (days since last birthday)
        last_birthday = date1.replace(year=date2.year)
        if date2 < last_birthday:
            last_birthday = last_birthday.replace(year=date2.year - 1)
        delta_days = (date2 - last_birthday).days
        if age_in_years < 0 or age_in_years > 150:
            self.logger.error(
                f"""
                [age_from_dates] Expected age in range [0:150], but got
                 {age_in_years} for date of birth = {date1} and
                 discharge date = {date2}.
                """
            )
        return age_in_years, delta_days
