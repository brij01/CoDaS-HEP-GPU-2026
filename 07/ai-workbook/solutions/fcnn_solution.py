"""
Module 07 AI-workbook - worked solution.

The fix: normalise with the model's FIXED per-feature mean/istd (the statistics
the network was trained with), not the batch's own statistics. The GPU version
then matches the CPU reference AND is much faster on a large batch -- fast *and*
correct.

Run:  python solutions/fcnn_solution.py
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
    """Trusted CPU reference (NumPy)."""
    xn = (X - p["mean"]) * p["istd"]
    h = np.maximum(xn @ p["W1"].T + p["b1"], 0.0)
    y = h @ p["W2"] + p["b2"]
    return 1.0 / (1.0 + np.exp(-y))


def fcnn_correct(X, p):
    """Same math as the reference, vectorised for the GPU (xp = cupy or numpy)."""
    xn = (X - p["mean"]) * p["istd"]     # FIX: the model's fixed mean/istd
    h = xp.maximum(xn @ p["W1"].T + p["b1"], 0.0)
    y = h @ p["W2"] + p["b2"]
    return 1.0 / (1.0 + xp.exp(-y))


def main():
    n = 2_000_000
    model = make_model()
    X_np = make_inputs(n)

    t0 = time.perf_counter()
    ref = fcnn_reference(X_np, model)
    cpu_t = time.perf_counter() - t0

    p = {k: xp.asarray(v) for k, v in model.items()}
    X = xp.asarray(X_np)
    if GPU:
        cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    got = fcnn_correct(X, p)
    if GPU:
        cp.cuda.Stream.null.synchronize()
    gpu_t = time.perf_counter() - t0

    got_np = cp.asnumpy(got) if GPU else np.asarray(got)
    max_err = float(np.max(np.abs(got_np - ref)))
    tol = 1e-4

    print(f"backend        : {'GPU (CuPy)' if GPU else 'CPU (NumPy fallback)'}")
    print(f"events         : {n:,}")
    print(f"CPU time       : {cpu_t * 1e3:8.1f} ms")
    print(f"compute time   : {gpu_t * 1e3:8.1f} ms")
    if GPU and gpu_t > 0:
        print(f"speedup vs CPU : {cpu_t / gpu_t:8.1f}x")
    print(f"max abs error  : {max_err:.3e}  (tol {tol:.0e})")
    print("RESULT         :", "PASS" if max_err <= tol else "FAIL")


if __name__ == "__main__":
    main()
