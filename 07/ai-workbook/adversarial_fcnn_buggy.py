"""
Module 07 AI-workbook - adversarial example (fast-but-wrong GPU inference).

An AI was asked to make the Allen-style single-hidden-layer network
(4 inputs -> 8 ReLU -> 1 sigmoid) run fast over a big batch of events on the GPU.
The version below runs, returns the right shape, and every output is a valid
probability in [0, 1] -- but it is SILENTLY WRONG.

The bug: to avoid uploading the trained per-feature mean/std, it "optimized" the
normalisation to use the *batch's own* statistics (mean/std computed over the
events in the batch). That is fast and shape-preserving, but it changes the math:
inference must use the model's fixed, trained mean/std, not statistics of whatever
batch happens to arrive.

Run it:  python adversarial_fcnn_buggy.py
Then use verify_fcnn.py to prove (against a CPU reference) that it is wrong.
"""
import numpy as np

try:
    import cupy as cp
    xp = cp
    BACKEND = "GPU (CuPy)"
except Exception:
    xp = np
    BACKEND = "CPU (NumPy fallback)"

NUM_INPUT, NUM_NODE = 4, 8


def make_model(seed=12345):
    """Reproducible model parameters (same seed everywhere in this workbook)."""
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


def fcnn_gpu_buggy(X, p):
    """Fast batched inference -- but normalises with BATCH statistics (the bug)."""
    mean = X.mean(axis=0)          # BUG: batch mean, not the model's fixed mean
    std = X.std(axis=0)            # BUG: batch std, not the model's fixed std
    xn = (X - mean) / std
    h = xp.maximum(xn @ p["W1"].T + p["b1"], 0.0)   # ReLU hidden layer
    y = h @ p["W2"] + p["b2"]
    return 1.0 / (1.0 + xp.exp(-y))                 # sigmoid output


if __name__ == "__main__":
    model = make_model()
    X = make_inputs(6)
    p = {k: xp.asarray(v) for k, v in model.items()}
    out = fcnn_gpu_buggy(xp.asarray(X), p)
    out_np = cp.asnumpy(out) if xp is not np else np.asarray(out)

    print(f"backend      : {BACKEND}")
    print(f"output shape : {out_np.shape}  (expected ({len(X)},))")
    print(f"all in [0, 1]: {bool((out_np >= 0).all() and (out_np <= 1).all())}")
    print("first outputs:", np.round(out_np, 4))
    print("\nLooks like valid probabilities with the right shape -- but is it correct?")
    print("Run verify_fcnn.py to compare it against a CPU reference.")
