"""ADVERSARIAL EXAMPLE -- this batched softmax "an AI wrote for you" is
INTENTIONALLY BUGGY. It runs, returns the right shape (batch, classes), and every
value is in [0, 1], but it normalises over the WRONG axis: down the batch
(axis=0) instead of across the classes (axis=1). The rows do not sum to 1, and
the predicted class (argmax per row) can be wrong.

It also skips the max-subtraction that keeps exp() from overflowing -- a real GPU
concern when logits are large.

Run it:
    python adversarial_softmax_buggy.py
"""

import numpy as np


def softmax_buggy(logits):
    e = np.exp(logits)                     # BUG: no max-subtraction (overflow risk)
    return e / e.sum(axis=0, keepdims=True)  # BUG: axis=0 normalises across the batch


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    logits = rng.normal(0.0, 2.0, size=(6, 4))   # 6 events, 4 classes

    probs = softmax_buggy(logits)
    row_sums = probs.sum(axis=1)
    print("per-row sums (should all be 1):")
    print(np.round(row_sums, 4))
