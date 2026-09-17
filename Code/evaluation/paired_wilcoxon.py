"""Query-level paired Wilcoxon + bootstrap CI (same protocol as paper analyses)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DIFF_ROUND_DECIMALS = 10
N_BOOT = 2000
BOOT_SEED = 42
ALPHA = 0.05


def paired_test(
    query_df: pd.DataFrame,
    col_a: str,
    col_b: str,
    name_a: str,
    name_b: str,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    alpha: float = ALPHA,
) -> dict:
    a = query_df[col_a].to_numpy(dtype=float)
    b = query_df[col_b].to_numpy(dtype=float)
    diffs = np.round(b - a, DIFF_ROUND_DECIMALS)
    n = len(diffs)
    n_pos = int((diffs > 0).sum())
    n_neg = int((diffs < 0).sum())
    n_zero = int((diffs == 0).sum())
    if n_zero == n:
        w_stat, p_value = float("nan"), 1.0
    else:
        res = wilcoxon(
            diffs, zero_method="wilcox", alternative="two-sided", method="auto"
        )
        w_stat, p_value = float(res.statistic), float(res.pvalue)

    observed = float(np.mean(b) - np.mean(a))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(b[idx]) - np.mean(a[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        "comparison": f"{name_b} vs {name_a}",
        "experimental_unit": "query",
        "n_query_pairs": n,
        "n_zero_differences_ties": n_zero,
        "n_b_higher": n_pos,
        "n_a_higher": n_neg,
        "Wilcoxon_statistic": w_stat,
        "p_value_two_sided": p_value,
        "significant_at_alpha_0.05": bool(p_value < alpha) if np.isfinite(p_value) else False,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_paired_difference_pp": observed * 100.0,
        "bootstrap": {
            "n_bootstrap": n_boot,
            "random_seed": seed,
            "ci_method": "percentile",
            "ci95_low_pp": float(lo) * 100.0,
            "ci95_high_pp": float(hi) * 100.0,
        },
    }
