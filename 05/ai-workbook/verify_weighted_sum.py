"""VERIFICATION HARNESS -- Phase 4 for Module 05.

This is YOUR test harness. It uses a high-precision reference (math.fsum, which
is exact for float64 inputs) and a relative-error PASS/FAIL gate. Implement the
function under test and run.

    python verify_weighted_sum.py
"""

import math
import numpy as np


def weighted_sum_reference(weights):
    """Source of truth: exact summation of the float64 values."""
    return math.fsum(weights.tolist())


def weighted_sum_under_test(weights):
    """TODO (student): implement (or paste the AI's) reduction.

    The default below is the buggy naive float32 accumulation and will FAIL.
    A correct fix accumulates in float64:
        return float(np.sum(weights, dtype=np.float64))
    """
    w32 = weights.astype(np.float32)
    total = np.float32(0.0)
    for w in w32:                          # naive sequential float32 -> precision loss
        total = np.float32(total + w)
    return float(total)


def main():
    rng = np.random.default_rng(1)
    weights = rng.uniform(0.0, 1e-3, size=2_000_000).astype(np.float64)

    ref = weighted_sum_reference(weights)
    got = weighted_sum_under_test(weights)
    rel = abs(got - ref) / abs(ref)
    tol = 1e-6
    print(f"reference (fsum): {ref:.6f}")
    print(f"under test      : {got:.6f}")
    print(f"relative error  : {rel:.3e} (tol {tol:.1e})")
    print("RESULT:", "PASS" if rel <= tol else "FAIL")


if __name__ == "__main__":
    main()
