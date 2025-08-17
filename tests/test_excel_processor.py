import os
import sys

# Add src/ to the module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from core.excel_processor import ExcelProcessor

def test_parse():
    file_path = os.path.abspath("data/sample/Customer_Ledger_Entries_FULL.xlsx")
    processor = ExcelProcessor(file_path)
    processor.parse()
    data = processor.get_data()
    assert isinstance(data, list)