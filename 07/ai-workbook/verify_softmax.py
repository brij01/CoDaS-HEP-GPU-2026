"""VERIFICATION HARNESS -- Phase 4 for Module 07.

This is YOUR test harness. It ships with a numerically stable reference softmax
and two gates: every row sums to 1, and element-wise agreement with the
reference. Implement the function under test and run.

    python verify_softmax.py
"""

import numpy as np


def softmax_reference(logits):
    """Source of truth: stable softmax across classes (axis=1)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def softmax_under_test(logits):
    """TODO (student): implement (or paste the AI's) softmax.

    Correct: subtract the per-row max, exp, divide by the per-row sum (axis=1).
    The default below is the buggy version and will FAIL.
    """
    e = np.exp(logits)
    return e / e.sum(axis=0, keepdims=True)   # buggy default (wrong axis)


def main():
    rng = np.random.default_rng(3)
    logits = rng.normal(0.0, 2.0, size=(6, 4))

    ref = softmax_reference(logits)
    got = softmax_under_test(logits)

    rows_ok = np.allclose(got.sum(axis=1), 1.0, atol=1e-6)
    match_ok = np.allclose(got, ref, atol=1e-6)
    if rows_ok and match_ok:
        print("PASS: every row sums to 1 and matches the reference softmax")
    else:
        print(f"FAIL: rows-sum-to-1 = {rows_ok}, matches-reference = {match_ok}")
        print("row sums:", np.round(got.sum(axis=1), 4))


if __name__ == "__main__":
    main()
