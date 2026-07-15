"""VERIFICATION HARNESS -- Phase 4 for Module 04.

This is YOUR test harness. It ships with a CPU reference (np.bincount) and a
bin-by-bin PASS/FAIL gate. Import the function under test, run it, and compare.

    python verify_histogram.py
"""

import numpy as np


def histogram_reference(values, edges):
    """CPU reference: the source of truth. Counts every event exactly once."""
    idx = np.digitize(values, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return np.bincount(idx, minlength=len(edges) - 1).astype(np.int64)


def histogram_under_test(values, edges):
    """TODO (student): implement (or paste the AI's) vectorised histogram.

    Correct options: np.add.at(hist, idx, 1) or np.bincount(idx, ...).
    On a GPU the equivalent is cuda.atomic.add(hist, idx, 1).
    """
    idx = np.digitize(values, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    hist = np.zeros(len(edges) - 1, dtype=np.int64)
    # np.add.at(hist, idx, 1)   # <-- uncomment for the correct accumulation
    return hist


def main():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, size=1_000_000)
    edges = np.linspace(-5.0, 5.0, 51)

    ref = histogram_reference(values, edges)
    got = histogram_under_test(values, edges)

    total_ok = int(got.sum()) == values.size
    bins_ok = np.array_equal(got, ref)
    if bins_ok and total_ok:
        print(f"PASS: all {len(ref)} bins match; total {got.sum()} == {values.size} events")
    else:
        bad = int(np.argmax(got != ref)) if not bins_ok else -1
        print(f"FAIL: total counted {int(got.sum())} of {values.size}; "
              f"first wrong bin {bad} (got {got[bad] if bad >= 0 else '-'}, "
              f"expected {ref[bad] if bad >= 0 else '-'})")


if __name__ == "__main__":
    main()
