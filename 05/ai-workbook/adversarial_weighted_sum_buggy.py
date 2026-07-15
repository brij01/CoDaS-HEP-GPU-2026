"""ADVERSARIAL EXAMPLE -- this weighted-sum reduction "an AI wrote for you" is
INTENTIONALLY BUGGY. It runs and returns a plausible number, but it accumulates
in float32, so it loses precision as the running total grows. This mirrors a GPU
reduction that uses the default single precision without care.

Run it:
    python adversarial_weighted_sum_buggy.py
"""

import numpy as np


def weighted_sum_buggy(weights):
    # BUG: cast to float32 and accumulate in float32. Low-order bits of each
    # addend are lost once the running sum is large compared to the addend.
    w32 = weights.astype(np.float32)
    total = np.float32(0.0)
    for w in w32:
        total = np.float32(total + w)      # explicit float32 accumulation
    return float(total)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    # Many small positive weights -> the classic precision-loss scenario.
    weights = rng.uniform(0.0, 1e-3, size=2_000_000).astype(np.float64)

    approx = weighted_sum_buggy(weights)
    exact = float(np.sum(weights, dtype=np.float64))
    print(f"float32 accumulation : {approx:.6f}")
    print(f"float64 reference    : {exact:.6f}")
    print(f"relative error       : {abs(approx - exact) / exact:.3e}")
