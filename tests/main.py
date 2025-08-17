import os
import sys
import time
from pathlib import Path

# --- Define current_dir first ---
current_dir = Path(__file__).parent

# --- Set up data paths relative to project root ---
project_root = current_dir.parent  # go up one level (out of tests/)
data_dir = project_root / "data" / "sample"
bank_file = data_dir / "KH_Bank.xlsx"
ledger_file = data_dir / "Customer_Ledger_Entries_FULL.xlsx"

try:
    from core.reconciliation.core import FinancialReconciler
    from core.reconciliation.visualization import plot_performance, plot_matches_by_method
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure:")
    print("1. You have a 'src' directory containing the modules")
    print("2. The directory structure matches the expected layout")
    sys.exit(1)

def main():
    # Initialize reconciler
    reconciler = FinancialReconciler()
    
    # ✅ Use already defined paths
    if not bank_file.exists() or not ledger_file.exists():
        print("Error: Data files not found!")
        print(f"Looking for: {bank_file} and {ledger_file}")
        print("Please ensure the data files exist in the correct location")
        return
