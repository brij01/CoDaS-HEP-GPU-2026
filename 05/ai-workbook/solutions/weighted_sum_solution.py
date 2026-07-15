"""SOLUTION -- correct weighted-sum reduction. Accumulate in float64 (and, for
the strictest case, use compensated/pairwise summation). Includes the GPU analog
note.

    python weighted_sum_solution.py
"""

import math
import numpy as np


def weighted_sum_reference(weights):
    return math.fsum(weights.tolist())


def weighted_sum_correct(weights):
    """float64 accumulation. NumPy's np.sum already uses pairwise summation,
    which keeps the error tiny; being explicit about dtype is the key fix."""
    return float(np.sum(weights, dtype=np.float64))


def weighted_sum_kahan(weights):
    """Kahan compensated summation -- the algorithm you'd use in a GPU reduction
    when you must accumulate in a lower precision. Shown for reference."""
    total = 0.0
    comp = 0.0
    for w in weights:
        y = float(w) - comp
        t = total + y
        comp = (t - total) - y
        total = t
    return total


def main():
    rng = np.random.default_rng(1)
    weights = rng.uniform(0.0, 1e-3, size=2_000_000).astype(np.float64)

    ref = weighted_sum_reference(weights)
    for name, fn in (("float64 sum", weighted_sum_correct),
                     ("kahan", weighted_sum_kahan)):
        got = fn(weights)
        rel = abs(got - ref) / abs(ref)
        print(f"{name:12s}: rel error {rel:.3e} -> "
              f"{'PASS' if rel <= 1e-6 else 'FAIL'}")


if __name__ == "__main__":
    main()
