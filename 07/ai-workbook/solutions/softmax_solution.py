"""SOLUTION -- correct, numerically stable batched softmax, with the CuPy (GPU)
drop-in shown for reference.

The bug was normalising over axis=0 (across the batch) and skipping the
max-subtraction. The fix reduces over axis=1 (across classes) after subtracting
the per-row max.

    python softmax_solution.py
"""

import numpy as np


def softmax_reference(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def softmax_correct(logits):
    """Stable softmax across classes."""
    shifted = logits - logits.max(axis=1, keepdims=True)   # overflow-safe
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)                 # reduce over classes


def softmax_gpu(logits):
    """Same code on the GPU with CuPy -- identical axis semantics. Runs only if
    CuPy and a CUDA device are present."""
    import cupy as cp

    x = cp.asarray(logits)
    shifted = x - x.max(axis=1, keepdims=True)
    e = cp.exp(shifted)
    out = e / e.sum(axis=1, keepdims=True)
    return cp.asnumpy(out)


def main():
    rng = np.random.default_rng(3)
    logits = rng.normal(0.0, 2.0, size=(6, 4))

    ref = softmax_reference(logits)
    cpu = softmax_correct(logits)
    rows_ok = np.allclose(cpu.sum(axis=1), 1.0)
    print("CPU fix:", "PASS" if rows_ok and np.allclose(cpu, ref) else "FAIL")

    try:
        gpu = softmax_gpu(logits)
        print("GPU (CuPy):", "PASS" if np.allclose(gpu, ref, atol=1e-6) else "FAIL")
    except Exception as exc:  # noqa: BLE001 -- no GPU / cupy not installed
        print(f"GPU check skipped (no CUDA device or cupy): {exc}")


if __name__ == "__main__":
    main()
