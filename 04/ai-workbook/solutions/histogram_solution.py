"""SOLUTION -- correct vectorised histogram, plus the GPU (Numba CUDA) analog.

The bug in adversarial_histogram_buggy.py was `hist[idx] += 1`, which does not
accumulate duplicate indices. The fix is an accumulating scatter-add:
`np.add.at` (CPU) or `cuda.atomic.add` (GPU).

    python histogram_solution.py
"""

import numpy as np


def histogram_reference(values, edges):
    idx = np.digitize(values, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return np.bincount(idx, minlength=len(edges) - 1).astype(np.int64)


def histogram_correct(values, edges):
    """CPU fix: np.add.at performs an unbuffered, accumulating scatter-add."""
    idx = np.digitize(values, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    hist = np.zeros(len(edges) - 1, dtype=np.int64)
    np.add.at(hist, idx, 1)            # accumulates every duplicate index
    return hist


# --- GPU analog (runs only if numba + a CUDA device are available) ------------
def histogram_gpu(values, edges):
    """The same fix on the GPU: every thread uses cuda.atomic.add into the bin."""
    from numba import cuda

    nbins = len(edges) - 1
    lo, hi = float(edges[0]), float(edges[-1])
    width = (hi - lo) / nbins

    @cuda.jit
    def _hist_kernel(vals, hist, lo, width, nbins):
        i = cuda.grid(1)
        if i < vals.size:
            b = int((vals[i] - lo) / width)
            if b < 0:
                b = 0
            elif b >= nbins:
                b = nbins - 1
            cuda.atomic.add(hist, b, 1)   # atomic: no lost updates

    d_vals = cuda.to_device(values)
    d_hist = cuda.to_device(np.zeros(nbins, dtype=np.int64))
    threads = 256
    blocks = (values.size + threads - 1) // threads
    _hist_kernel[blocks, threads](d_vals, d_hist, lo, width, nbins)
    return d_hist.copy_to_host()


def main():
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, size=1_000_000)
    edges = np.linspace(-5.0, 5.0, 51)

    ref = histogram_reference(values, edges)
    cpu = histogram_correct(values, edges)
    print("CPU fix:", "PASS" if np.array_equal(cpu, ref) and cpu.sum() == values.size else "FAIL")

    try:
        gpu = histogram_gpu(values, edges)
        print("GPU (atomic):", "PASS" if np.array_equal(gpu, ref) else "FAIL")
    except Exception as exc:  # noqa: BLE001 -- no GPU / numba not installed
        print(f"GPU check skipped (no CUDA device or numba): {exc}")


if __name__ == "__main__":
    main()
