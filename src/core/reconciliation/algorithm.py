import pandas as pd
import numpy as np
from itertools import combinations
from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt
import time

# src/core/reconciliation/algorithms.py

class ReconciliationAlgorithms:
    @staticmethod
    def direct(bank_data, ledger_data):
        # implement direct matching logic
        return []

    @staticmethod
    def subset(bank_data, ledger_data):
        # implement subset sum matching
        return []

    @staticmethod
    def dp(bank_data, ledger_data):
        # implement dynamic programming based reconciliation
        return []

    @staticmethod
    def fuzzy(bank_data, ledger_data):
        # implement fuzzy matching (string similarity etc.)
        return []



@dataclass
class MatchResult:
    transaction_ids: List[int]
    target_amount: float
    matched_amount: float
    confidence_score: float
    method: str

class FinancialReconciler:
    def __init__(self):
        self.transactions = None
        self.targets = None
        self.matches = []
        self.dropped_transactions = pd.DataFrame()
        self.dropped_targets = pd.DataFrame()
        
    def load_data(self, transaction_file: str, target_file: str):
        """Load and prepare data from Excel files"""
        # Load transaction data (Sheet 1)
        trans_df = pd.read_excel(transaction_file)
        self.transactions = pd.DataFrame({
            'Amount': pd.to_numeric(trans_df.iloc[:, 0], errors='coerce'),  # Column A
            'Description': trans_df.iloc[:, 1],                             # Column B
        })
        self.transactions['Transaction_ID'] = range(1, len(self.transactions) + 1)
        
        # Load target data (Sheet 2)
        target_df = pd.read_excel(target_file)
        self.targets = pd.DataFrame({
            'Target_Amount': pd.to_numeric(target_df.iloc[:, 2], errors='coerce'),  # Column C
            'Reference_ID': target_df.iloc[:, 3],                                   # Column D
        })
        self.targets['Target_ID'] = range(1, len(self.targets) + 1)
        
        # Clean data
        self._clean_data()
        
    def _clean_data(self):
        """Clean and standardize data (log dropped rows)"""
        # Log dropped rows
        self.dropped_transactions = self.transactions[self.transactions['Amount'].isna()]
        self.dropped_targets = self.targets[self.targets['Target_Amount'].isna()]
        
        # Drop NaNs
        self.transactions = self.transactions.dropna(subset=['Amount'])
        self.targets = self.targets.dropna(subset=['Target_Amount'])
        
        # Convert to float
        self.transactions['Amount'] = self.transactions['Amount'].astype(float)
        self.targets['Target_Amount'] = self.targets['Target_Amount'].astype(float)

    def direct_matching(self, tolerance: float = 0.01) -> List[MatchResult]:
        """Direct matching between individual transactions and targets"""
        matches = []
        for _, target_row in self.targets.iterrows():
            target_amount = target_row['Target_Amount']
            relative_tolerance = max(tolerance, target_amount * 0.01)
            
            for _, trans_row in self.transactions.iterrows():
                if abs(trans_row['Amount'] - target_amount) <= relative_tolerance:
                    matches.append(MatchResult(
                        transaction_ids=[trans_row['Transaction_ID']],
                        target_amount=target_amount,
                        matched_amount=trans_row['Amount'],
                        confidence_score=1.0 - (abs(trans_row['Amount'] - target_amount) / target_amount),
                        method="Direct Match"
                    ))
        return matches

    def subset_sum_brute_force(self, max_combo_size: int = 3, tolerance: float = 0.01) -> List[MatchResult]:
        """Subset sum brute force solution"""
        matches = []
        trans_records = self.transactions.to_dict('records')
        
        for target in self.targets.to_dict('records'):
            target_amount = target['Target_Amount']
            
            for combo_size in range(2, min(max_combo_size + 1, len(trans_records))):
                for combo in combinations(trans_records, combo_size):
                    combo_sum = sum(t['Amount'] for t in combo)
                    
                    if abs(combo_sum - target_amount) <= tolerance:
                        matches.append(MatchResult(
                            transaction_ids=[t['Transaction_ID'] for t in combo],
                            target_amount=target_amount,
                            matched_amount=combo_sum,
                            confidence_score=1.0 - (abs(combo_sum - target_amount) / target_amount),
                            method=f"Subset Sum (size {combo_size})"
                        ))
        return matches

    def dynamic_programming_subset(self, tolerance: float = 0.01) -> List[MatchResult]:
        """Dynamic programming subset sum solution"""
        matches = []
        amounts = self.transactions['Amount'].values
        target_amounts = self.targets['Target_Amount'].values
        
        def is_subset_sum_possible(arr, target):
            target_int = int(target * 100)
            arr_int = [int(x * 100) for x in arr]
            dp = [False] * (target_int + 1)
            dp[0] = True
            
            for num in arr_int:
                for j in range(target_int, num - 1, -1):
                    if dp[j - num]:
                        dp[j] = True
            
            tolerance_int = int(target * tolerance * 100)
            for amount in range(max(0, target_int - tolerance_int), 
                             min(target_int + tolerance_int + 1, len(dp))):
                if dp[amount]:
                    return True
            return False
        
        for target in target_amounts:
            if is_subset_sum_possible(amounts, target):
                matches.append(MatchResult(
                    transaction_ids=[],
                    target_amount=target,
                    matched_amount=target,
                    confidence_score=0.9,
                    method="Dynamic Programming"
                ))
        return matches

    def fuzzy_matching(self, amount_weight: float = 0.8, min_similarity: float = 0.7):
        """Fuzzy matching with similarity scores"""
        matches = []
        
        for _, trans_row in self.transactions.iterrows():
            trans_amount = trans_row['Amount']
            trans_desc = str(trans_row['Description'])
            
            for _, target_row in self.targets.iterrows():
                target_amount = target_row['Target_Amount']
                target_ref = str(target_row['Reference_ID'])
                
                amount_sim = 1.0 / (1.0 + abs(trans_amount - target_amount) / max(trans_amount, target_amount))
                text_sim = 0.5  # Simplified text similarity
                combined_sim = amount_weight * amount_sim + (1 - amount_weight) * text_sim
                
                if combined_sim >= min_similarity:
                    matches.append(MatchResult(
                        transaction_ids=[trans_row['Transaction_ID']],
                        target_amount=target_amount,
                        matched_amount=trans_amount,
                        confidence_score=combined_sim,
                        method="Fuzzy Matching"
                    ))
        return matches

    def reconcile(self, methods: List[str] = ['direct', 'subset', 'dp', 'fuzzy']):
        """Run reconciliation using specified methods"""
        self.matches = []
        
        if 'direct' in methods:
            self.matches.extend(self.direct_matching())
        if 'subset' in methods:
            self.matches.extend(self.subset_sum_brute_force())
        if 'dp' in methods:
            self.matches.extend(self.dynamic_programming_subset())
        if 'fuzzy' in methods:
            self.matches.extend(self.fuzzy_matching())
            
        return self.matches
    
    def generate_report(self, output_file: str):
        """Generate Excel report with reconciliation results"""
        report_data = []
        
        for match in self.matches:
            report_data.append({
                'Method': match.method,
                'Target Amount': match.target_amount,
                'Matched Amount': match.matched_amount,
                'Confidence Score': match.confidence_score,
                'Transaction IDs': ', '.join(map(str, match.transaction_ids))
            })
        
        report_df = pd.DataFrame(report_data)
        
        with pd.ExcelWriter(output_file) as writer:
            report_df.to_excel(writer, sheet_name='Reconciliation Results', index=False)
            self.transactions.to_excel(writer, sheet_name='Transactions', index=False)
            self.targets.to_excel(writer, sheet_name='Targets', index=False)
            self.dropped_transactions.to_excel(writer, sheet_name='Dropped Transactions', index=False)
            self.dropped_targets.to_excel(writer, sheet_name='Dropped Targets', index=False)

def plot_performance(execution_times: Dict[str, float]):
    """Plot performance comparison"""
    plt.figure(figsize=(10, 6))
    plt.bar(execution_times.keys(), execution_times.values())
    plt.title('Algorithm Performance Comparison')
    plt.ylabel('Execution Time (seconds)')
    plt.xlabel('Matching Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.close()

def plot_matches_by_method(matches: List[MatchResult]):
    """Plot distribution of matches by method"""
    method_counts = {}
    for match in matches:
        method = match.method.split('(')[0].strip()
        method_counts[method] = method_counts.get(method, 0) + 1
    
    plt.figure(figsize=(10, 6))
    plt.bar(method_counts.keys(), method_counts.values())
    plt.title('Matches Found by Method')
    plt.ylabel('Number of Matches')
    plt.xlabel('Matching Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('matches_by_method.png')
    plt.close()

def main():
    """Main execution function"""
    # Initialize reconciler
    reconciler = FinancialReconciler()
    
    # Load data (update these paths as needed)
    reconciler.load_data('data/sample/KH_Bank.xlsx', 'data/sample/Customer_Ledger_Entries_FULL.xlsx')
    
    # Run reconciliation with timing
    methods = ['direct', 'subset', 'dp', 'fuzzy']
    exec_times = {}
    
    for method in methods:
        start_time = time.time()
        reconciler.reconcile(methods=[method])
        exec_times[method] = time.time() - start_time
    
    # Run full reconciliation
    reconciler.reconcile(methods=methods)
    
    # Generate outputs
    reconciler.generate_report('Financial_Reconciliation_Report.xlsx')
    plot_performance(execution_times=exec_times)
    plot_matches_by_method(reconciler.matches)
    
    print("Reconciliation complete! Results saved to:")
    print("- Financial_Reconciliation_Report.xlsx")
    print("- performance_comparison.png")
    print("- matches_by_method.png")

if __name__ == "__main__":
    main()
