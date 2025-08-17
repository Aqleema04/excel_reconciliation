import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import sys, os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict


# ✅ Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

#from src.core.reconciliation.core import MatchResult

# ✅ Flexible import: absolute first, then relative
try:
    from src.core.reconciliation.core import MatchResult  # when run directly
except ImportError:
   from .core import MatchResult  # when used as part of package


def plot_performance(execution_times: Dict[str, float]):
    """Plot performance comparison of different methods"""
    methods = list(execution_times.keys())
    times = list(execution_times.values())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=methods, y=times)
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
    sns.barplot(x=list(method_counts.keys()), y=list(method_counts.values()))
    plt.title('Matches Found by Method')
    plt.ylabel('Number of Matches')
    plt.xlabel('Matching Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('matches_by_method.png')
    plt.close()


# ✅ Optional: allow testing directly
if __name__ == "__main__":
    from collections import namedtuple
    
    # Fake matches for demo
    FakeMatch = namedtuple("FakeMatch", ["method"])
    sample_matches = [FakeMatch("direct"), FakeMatch("direct"),
                      FakeMatch("fuzzy"), FakeMatch("dp")]
    
    plot_performance({"direct": 0.1, "fuzzy": 0.25, "dp": 0.4})
    plot_matches_by_method(sample_matches)
    print("✅ Test plots saved (performance_comparison.png, matches_by_method.png)")
