from src.core.reconciliation.core import FinancialReconciler
import sys, time, json, logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ReconciliationBenchmark:
    """Quick benchmark wrapper for FinancialReconciler"""

    def __init__(self) -> None:
        self.results: Dict[str, Dict] = {
            "execution_times": {},
            "accuracy_metrics": {},
            "scalability": {},
        }
        self.reconciler = FinancialReconciler()

        # ✅ Smaller sizes for fast execution
        self.dataset_sizes = [50, 100, 200]
        self.method_keys = ["direct", "subset", "dp", "fuzzy"]

    # ---------------- Data ----------------
    def generate_test_data(self, num_transactions: int, num_targets: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(42 + num_transactions)

        transactions = pd.DataFrame({
            "Transaction_ID": np.arange(1, num_transactions + 1),
            "Amount": rng.uniform(10, 500, size=num_transactions).round(2),
            "Description": [f"TRANS_{i}" for i in range(num_transactions)],
        })

        targets = []
        for i in range(min(num_targets, max(2, num_transactions // 4))):
            subset = transactions.sample(n=2, random_state=int(rng.integers(1_000_000)))
            targets.append({
                "Target_Amount": round(subset["Amount"].sum(), 2),
                "Reference_ID": f"REF_{i}",
                "Target_ID": i + 1
            })

        return transactions, pd.DataFrame(targets)

    # ---------------- Dispatcher ----------------
    def _run_method(self, method_key: str, transactions: pd.DataFrame, targets: pd.DataFrame):
        self.reconciler.transactions = transactions.copy()
        self.reconciler.targets = targets.copy()
        return self.reconciler.reconcile(methods=[method_key])

    # ---------------- Timing ----------------
    def time_method(self, method_key: str, tx: pd.DataFrame, tg: pd.DataFrame) -> float:
        start = time.perf_counter()
        _ = self._run_method(method_key, tx, tg)
        return time.perf_counter() - start

    # ---------------- Accuracy ----------------
    def test_accuracy(self, tx: pd.DataFrame, tg: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        known_amounts = set(float(a) for a in tg["Target_Amount"].tolist())
        results: Dict[str, Dict[str, float]] = {}

        for key in self.method_keys:
            if key == "subset" and len(tx) > 50:  
                results[key] = {"precision": 0, "recall": 0, "f1_score": 0}
                continue

            try:
                matches = self._run_method(key, tx, tg)
            except Exception as e:
                logger.warning("Method %s failed: %s", key, e)
                results[key] = {"precision": 0, "recall": 0, "f1_score": 0}
                continue

            tp, fp = 0, 0
            for m in matches:
                md = asdict(m) if is_dataclass(m) else dict(m)
                matched = any(abs(md.get("target_amount", 0) - ka) < 0.01 for ka in known_amounts)
                tp += int(matched)
                fp += int(not matched)

            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / len(known_amounts) if known_amounts else 0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0

            results[key] = {"precision": round(prec, 3),
                            "recall": round(rec, 3),
                            "f1_score": round(f1, 3)}
        return results

    # ---------------- Benchmark Suites ----------------
    def run_scalability_test(self) -> None:
        logger.info("▶ Scalability test...")
        for size in self.dataset_sizes:
            tx, tg = self.generate_test_data(size, size // 2)
            times: Dict[str, float | None] = {}
            for key in self.method_keys:
                if key == "subset" and size > 50:  
                    times[key] = None
                    continue
                avg = self.time_method(key, tx, tg)
                times[key] = round(avg, 4)
            self.results["scalability"][size] = times

    def run_accuracy_test(self) -> None:
        logger.info("▶ Accuracy test...")
        tx, tg = self.generate_test_data(80, 20)
        self.results["accuracy_metrics"] = self.test_accuracy(tx, tg)

    def run_performance_test(self) -> None:
        logger.info("▶ Performance test...")
        tx, tg = self.generate_test_data(100, 30)
        for key in self.method_keys:
            if key == "subset":  
                continue
            avg = self.time_method(key, tx, tg)
            self.results["execution_times"][key] = round(avg, 4)

    # ---------------- Output ----------------
    def generate_plots(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Execution times
        plt.figure(figsize=(6, 4))
        plt.bar(self.results["execution_times"].keys(),
                self.results["execution_times"].values())
        plt.title("Execution Times")
        plt.ylabel("Seconds")
        plt.tight_layout()
        plt.savefig(output_dir / "execution_times.png")
        plt.close()

        # Scalability
        plt.figure(figsize=(6, 4))
        sizes = list(self.results["scalability"].keys())
        for key in self.method_keys:
            y = [self.results["scalability"][s].get(key) for s in sizes]
            if any(v is not None for v in y):
                plt.plot(sizes, y, marker="o", label=key)
        plt.title("Scalability")
        plt.xlabel("Transactions")
        plt.ylabel("Seconds")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "scalability.png")
        plt.close()

        # Accuracy
        for metric in ["precision", "recall", "f1_score"]:
            plt.figure(figsize=(6, 4))
            vals = [self.results["accuracy_metrics"][k][metric]
                    for k in self.method_keys]
            plt.bar(self.method_keys, vals)
            plt.title(metric.capitalize())
            plt.tight_layout()
            plt.savefig(output_dir / f"{metric}.png")
            plt.close()

    def save_results(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        excel_path = output_dir / "benchmark_results.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame.from_dict(self.results["execution_times"], orient="index").to_excel(
                writer, sheet_name="Execution Times")
            pd.DataFrame(self.results["accuracy_metrics"]).T.to_excel(
                writer, sheet_name="Accuracy Metrics")
            pd.DataFrame(self.results["scalability"]).T.to_excel(
                writer, sheet_name="Scalability")

    def run_full_benchmark(self) -> None:
        self.run_performance_test()
        self.run_accuracy_test()
        self.run_scalability_test()
        out_dir = PROJECT_ROOT / "benchmark_results"
        self.generate_plots(out_dir)
        self.save_results(out_dir)
        logger.info("✅ Benchmark finished FAST! Results saved to %s", out_dir)


if __name__ == "__main__":
    bench = ReconciliationBenchmark()
    bench.run_full_benchmark()
