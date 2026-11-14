"""Statistical tools used to compare ranking methods (Part 3 extras).

* ``paired_permutation_test`` - is the difference between two methods,
  measured on the SAME queries, larger than chance? (randomisation test,
  the standard significance test in IR evaluation; Smucker et al., 2007)
* ``rank_biased_overlap`` - similarity of two rankings that gives more
  weight to the top positions (Webber et al., 2010). It complements
  Kendall's tau, which weights every position equally.
"""

from itertools import product

import numpy as np


def paired_permutation_test(a, b, n_resamples: int = 100_000, seed: int = 42):
    """Two-sided paired randomisation test on per-query scores.

    Under the null hypothesis the two methods are interchangeable, so the
    sign of each per-query difference is equally likely to be + or -.
    With few queries (n <= 16) all 2^n sign assignments are enumerated, so
    the p-value is exact; otherwise ``n_resamples`` random ones are used.

    Args:
        a: Per-query scores of method A (e.g. AP@10 for Q1..Q8).
        b: Per-query scores of method B, same queries in the same order.
        n_resamples: Random sign flips used when n > 16.
        seed: Random seed (reproducibility).

    Returns:
        ``(mean_difference, p_value)`` where the difference is mean(a - b).
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    observed = abs(diff.mean())
    if n == 0 or np.allclose(diff, 0):
        return 0.0, 1.0
    if n <= 16:
        signs = np.array(list(product((1, -1), repeat=n)))
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((1, -1), size=(n_resamples, n))
    null = np.abs((signs * diff).mean(axis=1))
    p_value = float(np.mean(null >= observed - 1e-12))
    return float(diff.mean()), p_value


def rank_biased_overlap(a: list, b: list, p: float = 0.9, depth: int = 20) -> float:
    """Rank-biased overlap (RBO) of two rankings, truncated at ``depth``.

    ``RBO = (1 - p) * sum_{d=1..depth} p^(d-1) * |A_d n B_d| / d``, where
    A_d is the top-d of ranking a. With p = 0.9 the top 10 positions carry
    about 86% of the weight. The truncated value is a lower bound in [0, 1).

    Args:
        a: First ranked list.
        b: Second ranked list.
        p: Persistence (higher = deeper user patience).
        depth: Evaluation depth.

    Returns:
        RBO in [0, 1); identical top-``depth`` lists give 1 - p^depth.
    """
    seen_a, seen_b, overlap, total = set(), set(), 0, 0.0
    for d in range(1, depth + 1):
        x = a[d - 1] if d <= len(a) else None
        y = b[d - 1] if d <= len(b) else None
        if x is not None:
            overlap += x in seen_b
            seen_a.add(x)
        if y is not None:
            overlap += y in seen_a
            seen_b.add(y)
        total += p ** (d - 1) * overlap / d
    return (1 - p) * total
