"""
Module 07 AI-workbook - VERIFY harness (Phase 4 of the Module 11 loop).

Edit fcnn_under_test() until RESULT is PASS: its output must match the CPU
reference within tolerance over a large batch of events. The harness also reports
the GPU-vs-CPU speedup, so you can see that "fast" and "correct" are two separate
questions -- the whole point of this workbook.

Run:  python verify_fcnn.py
"""
import time
import numpy as np

try:
    import cupy as cp
    xp = cp
    GPU = True
except Exception:
    xp = np
    GPU = False

NUM_INPUT, NUM_NODE = 4, 8


def make_model(seed=12345):
    rng = np.random.default_rng(seed)
    return {
        "mean": rng.uniform(-1, 1, NUM_INPUT).astype(np.float32),
        "istd": (1.0 / rng.uniform(0.5, 1.5, NUM_INPUT)).astype(np.float32),
        "W1": rng.uniform(-1, 1, (NUM_NODE, NUM_INPUT)).astype(np.float32),
        "b1": rng.uniform(-1, 1, NUM_NODE).astype(np.float32),
        "W2": rng.uniform(-1, 1, NUM_NODE).astype(np.float32),
        "b2": np.float32(rng.uniform(-1, 1)),
    }


def make_inputs(n, seed=2024):
    rng = np.random.default_rng(seed)
    return rng.uniform(-3, 3, (n, NUM_INPUT)).astype(np.float32)


def fcnn_reference(X, p):
    """Trusted CPU reference (NumPy). The GPU result must match this."""
    xn = (X - p["mean"]) * p["istd"]
    h = np.maximum(xn @ p["W1"].T + p["b1"], 0.0)
    y = h @ p["W2"] + p["b2"]
    return 1.0 / (1.0 + np.exp(-y))


def fcnn_under_test(X, p):
    # <<< EDIT THIS until RESULT is PASS >>>
    # Default = the buggy "AI" version: it normalises with BATCH statistics
    # instead of the model's fixed per-feature mean/istd. Fast, right shape,
    # every value in [0, 1] -- and wrong.
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    xn = (X - mean) / std
    h = xp.maximum(xn @ p["W1"].T + p["b1"], 0.0)
    y = h @ p["W2"] + p["b2"]
    return 1.0 / (1.0 + xp.exp(-y))


def main():
    n = 2_000_000
    model = make_model()
    X_np = make_inputs(n)

    # CPU reference (source of truth) + timing.
    t0 = time.perf_counter()
    ref = fcnn_reference(X_np, model)
    cpu_t = time.perf_counter() - t0

    # Version under test, on the GPU if CuPy is available.
    p = {k: xp.asarray(v) for k, v in model.items()}
    X = xp.asarray(X_np)
    if GPU:
        cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    got = fcnn_under_test(X, p)
    if GPU:
        cp.cuda.Stream.null.synchronize()
    test_t = time.perf_counter() - t0

    got_np = cp.asnumpy(got) if GPU else np.asarray(got)
    shape_ok = got_np.shape == ref.shape
    max_err = float(np.max(np.abs(got_np - ref))) if shape_ok else float("inf")
    tol = 1e-4
    ok = shape_ok and max_err <= tol

    print(f"backend under test : {'GPU (CuPy)' if GPU else 'CPU (NumPy fallback)'}")
    print(f"events             : {n:,}")
    print(f"CPU reference time : {cpu_t * 1e3:8.1f} ms")
    print(f"under-test time    : {test_t * 1e3:8.1f} ms")
    if GPU and test_t > 0:
        print(f"speedup vs CPU     : {cpu_t / test_t:8.1f}x")
    print(f"shape matches ref  : {shape_ok}")
    print(f"max abs error      : {max_err:.3e}  (tol {tol:.0e})")
    print("RESULT             :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
