"""SOLUTION -- correct ray-sphere intersection returning the nearest positive
hit. The bug was taking the far root (-b + sqrt) instead of the near root.

    python raytrace_solution.py
"""

import numpy as np


def intersect_reference(origins, dirs, center, radius):
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    oc = origins - center
    b = np.sum(oc * d, axis=1)
    c = np.sum(oc * oc, axis=1) - radius * radius
    disc = b * b - c
    hit = disc >= 0.0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t_near = -b - sq
    t_far = -b + sq
    t = np.where(t_near > 1e-6, t_near, t_far)
    return np.where(hit & (t > 1e-6), t, np.inf)


def intersect_correct(origins, dirs, center, radius):
    """Nearest positive root. Normalise the direction so t is a true distance."""
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    oc = origins - center
    b = np.sum(oc * d, axis=1)
    c = np.sum(oc * oc, axis=1) - radius * radius
    disc = b * b - c
    hit = disc >= 0.0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t_near = -b - sq                       # near root first
    t_far = -b + sq
    t = np.where(t_near > 1e-6, t_near, t_far)
    return np.where(hit & (t > 1e-6), t, np.inf)


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
    got = intersect_correct(origins, dirs, center, radius)
    ok = np.allclose(np.where(np.isfinite(ref), ref, 0.0),
                     np.where(np.isfinite(got), got, 0.0), atol=1e-6)
    print("CPU fix:", "PASS" if ok else "FAIL")
    print("nearest hit distances:", np.round(got, 3))


if __name__ == "__main__":
    main()
