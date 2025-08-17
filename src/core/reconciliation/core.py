from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np
from difflib import SequenceMatcher


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

    def load_data(self, transaction_file: str, target_file: str):
        trans_df = pd.read_excel(transaction_file)
        self.transactions = pd.DataFrame({
            "Amount": pd.to_numeric(trans_df.iloc[:, 0], errors="coerce"),
            "Description": trans_df.iloc[:, 1],
        })
        self.transactions["Transaction_ID"] = range(1, len(self.transactions) + 1)

        target_df = pd.read_excel(target_file)
        self.targets = pd.DataFrame({
            "Target_Amount": pd.to_numeric(target_df.iloc[:, 2], errors="coerce"),
            "Reference_ID": target_df.iloc[:, 3],
        })
        self.targets["Target_ID"] = range(1, len(self.targets) + 1)

        self._clean_data()

    def _clean_data(self):
        self.transactions = self.transactions.dropna(subset=["Amount"])
        self.targets = self.targets.dropna(subset=["Target_Amount"])
        self.transactions["Amount"] = self.transactions["Amount"].astype(float)
        self.targets["Target_Amount"] = self.targets["Target_Amount"].astype(float)

    # ----------------- FIXED reconcile -----------------
    def reconcile(self, methods: List[str] = ["direct", "subset", "dp", "fuzzy"]):
        """Run reconciliation using specified methods"""
        self.matches = []

        if "direct" in methods:
            self.matches.extend(self._direct_matching())
        if "subset" in methods:
            self.matches.extend(self._subset_sum_brute_force())
        if "dp" in methods:
            self.matches.extend(self._dynamic_programming_subset())
        if "fuzzy" in methods:
            self.matches.extend(self._fuzzy_matching())

        return self.matches

    # ----------------- Matching Algorithms -----------------
    def _direct_matching(self):
        results = []
        for _, t in self.targets.iterrows():
            exact = self.transactions[self.transactions["Amount"] == t["Target_Amount"]]
            for _, tx in exact.iterrows():
                results.append(
                    MatchResult(
                        transaction_ids=[tx["Transaction_ID"]],
                        target_amount=t["Target_Amount"],
                        matched_amount=tx["Amount"],
                        confidence_score=1.0,
                        method="direct",
                    )
                )
        return results

    def _subset_sum_brute_force(self):
        results = []
        tx_amounts = self.transactions[["Transaction_ID", "Amount"]].values
        for _, t in self.targets.iterrows():
            target = t["Target_Amount"]
            for i in range(len(tx_amounts)):
                for j in range(i + 1, len(tx_amounts)):
                    if abs(tx_amounts[i][1] + tx_amounts[j][1] - target) < 0.01:
                        results.append(
                            MatchResult(
                                transaction_ids=[int(tx_amounts[i][0]), int(tx_amounts[j][0])],
                                target_amount=target,
                                matched_amount=tx_amounts[i][1] + tx_amounts[j][1],
                                confidence_score=0.9,
                                method="subset",
                            )
                        )
        return results

    def _dynamic_programming_subset(self):
        results = []
        amounts = self.transactions["Amount"].tolist()
        ids = self.transactions["Transaction_ID"].tolist()
        n = len(amounts)

        for _, t in self.targets.iterrows():
            target = round(t["Target_Amount"], 2)
            dp = {0: []}
            for i in range(n):
                new_dp = dp.copy()
                for s, comb in dp.items():
                    new_sum = round(s + amounts[i], 2)
                    if new_sum not in new_dp:
                        new_dp[new_sum] = comb + [ids[i]]
                dp.update(new_dp)

            if target in dp:
                results.append(
                    MatchResult(
                        transaction_ids=dp[target],
                        target_amount=target,
                        matched_amount=target,
                        confidence_score=0.95,
                        method="dp",
                    )
                )
        return results

    def _fuzzy_matching(self):
        results = []
        for _, t in self.targets.iterrows():
            for _, tx in self.transactions.iterrows():
                ratio = SequenceMatcher(None, str(t["Reference_ID"]), str(tx["Description"])).ratio()
                if ratio > 0.8:  # threshold
                    results.append(
                        MatchResult(
                            transaction_ids=[tx["Transaction_ID"]],
                            target_amount=t["Target_Amount"],
                            matched_amount=tx["Amount"],
                            confidence_score=ratio,
                            method="fuzzy",
                        )
                    )
        return results
