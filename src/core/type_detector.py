
import re
from datetime import datetime


class TypeDetector:
    @staticmethod
    def detect_type(value):
        if isinstance(value, (int, float)):
            return "Number"
        elif isinstance(value, datetime):
            return "Date"
        elif isinstance(value, str):
            if re.match(r"\d{4}-\d{2}-\d{2}", value):
                return "Date"
            elif re.match(r"^\d+(\.\d+)?$", value):
                return "Number"
            return "Text"
        return "Unknown"

    def _detect_date_format(self, sample_values: list[str]) -> str:
        # Dummy logic: return format if sample matches ISO
        for val in sample_values:
            if re.match(r"\d{4}-\d{2}-\d{2}", val):
                return "YYYY-MM-DD"
        return "unknown"
