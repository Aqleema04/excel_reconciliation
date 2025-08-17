
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pandas as pd
from core.data_storage import DataStorage
from core.type_detector import TypeDetector  # Make sure TypeDetector has detect_type()

class ExcelProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.storage = DataStorage()

    def parse(self):
        df = pd.read_excel(self.file_path)
        for _, row in df.iterrows():
            parsed_row = {}
            for col, val in row.items():
                parsed_row[col] = {
                    "value": val,
                    "type": TypeDetector.detect_type(val)  # This must be a @staticmethod
                }
            self.storage.add_row(parsed_row)

    def get_data(self):
        return self.storage.get_all()

# ✅ Allow direct execution
if __name__ == "__main__":
    # Absolute path to the Excel file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(current_dir, "..", "..", "data", "sample", "Customer_Ledger_Entries_FULL.xlsx")
    excel_path = os.path.abspath(excel_path)  # Resolve path

    # Print to confirm
    print("Reading from:", excel_path)

    # Run the processor
    processor = ExcelProcessor(excel_path)
    processor.parse()

    data = processor.get_data()
    for i, row in enumerate(data[:5]):  # Print first 5 rows only
        print(f"\n--- Row {i + 1} ---")
        for col, details in row.items():
            print(f"{col}: {details}")
