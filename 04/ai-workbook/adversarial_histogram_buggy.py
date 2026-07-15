"""ADVERSARIAL EXAMPLE -- this histogram "an AI wrote for you" is INTENTIONALLY
BUGGY. It runs without error and its shape looks like a histogram, but it
UNDERCOUNTS every bin.

The bug: `hist[idx] += 1` on a NumPy array with repeated entries in `idx` is
NOT an accumulation. NumPy evaluates it as `hist[idx] = hist[idx] + 1`, so each
distinct index is incremented exactly once regardless of how many times it
appears. This is the CPU mirror of a GPU histogram that forgot `atomicAdd`:
concurrent "+= 1" into the same bin loses updates.

Run it:
    python adversarial_histogram_buggy.py
"""

import numpy as np


def histogram_buggy(values, edges):
    idx = np.digitize(values, edges) - 1          # bin index per event
    idx = np.clip(idx, 0, len(edges) - 2)
    hist = np.zeros(len(edges) - 1, dtype=np.int64)
    hist[idx] += 1                                 # BUG: does not accumulate duplicates
    return hist


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, size=1_000_000)
    edges = np.linspace(-5.0, 5.0, 51)

    hist = histogram_buggy(values, edges)
    print("total counted:", hist.sum(), "of", values.size, "events")
    print("-> undercount:", values.size - int(hist.sum()))
