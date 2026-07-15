"""VERIFICATION HARNESS -- Phase 4 for Module 09.

This is YOUR test harness. It ships with a reference ray-sphere intersection
(nearest positive root) and a per-ray comparison of t. Implement the function
under test and run.

    python verify_raytrace.py
"""

import numpy as np


def intersect_reference(origins, dirs, center, radius):
    """Source of truth: nearest positive intersection distance per ray."""
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    oc = origins - center
    b = np.sum(oc * d, axis=1)
    c = np.sum(oc * oc, axis=1) - radius * radius
    disc = b * b - c
    hit = disc >= 0.0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t_near = -b - sq
    t_far = -b + sq
    # nearest positive root
    t = np.where(t_near > 1e-6, t_near, t_far)
    t = np.where(hit & (t > 1e-6), t, np.inf)
    return t


def intersect_under_test(origins, dirs, center, radius):
    """TODO (student): implement (or paste the AI's) intersection.

    Correct: pick the smallest positive root (-b - sqrt(disc) when positive).
    The default below returns the far root and will FAIL.
    """
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    oc = origins - center
    b = np.sum(oc * d, axis=1)
    c = np.sum(oc * oc, axis=1) - radius * radius
    disc = b * b - c
    hit = disc >= 0.0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t = -b + sq                        # buggy default (far root)
    return np.where(hit & (t > 0.0), t, np.inf)


def main():
    rng = np.random.default_rng(5)
    n = 8
    origins = np.zeros((n, 3)); origins[:, 2] = -5.0
    dirs = np.zeros((n, 3))
    dirs[:, 0] = rng.uniform(-0.3, 0.3, size=n)
    dirs[:, 1] = rng.uniform(-0.3, 0.3, size=n)
    dirs[:, 2] = 1.0
    center = np.array([0.0, 0.0, 0.0]); radius = 1.0

    ref = intersect_reference(origins, dirs, center, radius)
    got = intersect_under_test(origins, dirs, center, radius)

    finite = np.isfinite(ref) | np.isfinite(got)
    ok = np.allclose(np.where(np.isfinite(ref), ref, 0.0),
                     np.where(np.isfinite(got), got, 0.0), atol=1e-4) and \
        np.array_equal(np.isfinite(ref), np.isfinite(got))
    if ok:
        print("PASS: hit distances match the reference for all rays")
    else:
        print("FAIL: hit distances differ")
        print("reference:", np.round(ref, 3))
        print("under test:", np.round(got, 3))
        _ = finite


if __name__ == "__main__":
    main()
