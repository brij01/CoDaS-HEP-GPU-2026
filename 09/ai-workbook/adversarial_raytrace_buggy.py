"""ADVERSARIAL EXAMPLE -- this ray-sphere intersection "an AI wrote for you" is
INTENTIONALLY BUGGY. It runs and reports hits for the correct pixels, but it
returns the FAR intersection (-b + sqrt(disc)) instead of the NEAR one
(-b - sqrt(disc)). Every hit distance is therefore the back of the sphere, so
depths, normals, and shading are all wrong -- even though a thumbnail still shows
a sphere-shaped blob.

Run it:
    python adversarial_raytrace_buggy.py
"""

import numpy as np


def intersect_buggy(origins, dirs, center, radius):
    d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    oc = origins - center
    b = np.sum(oc * d, axis=1)
    c = np.sum(oc * oc, axis=1) - radius * radius
    disc = b * b - c
    hit = disc >= 0.0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t = -b + sq                 # BUG: far root; should be -b - sq for nearest hit
    t = np.where(hit & (t > 0.0), t, np.inf)
    return t


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    n = 8
    origins = np.zeros((n, 3))
    origins[:, 2] = -5.0
    dirs = np.zeros((n, 3))
    dirs[:, 0] = rng.uniform(-0.3, 0.3, size=n)
    dirs[:, 1] = rng.uniform(-0.3, 0.3, size=n)
    dirs[:, 2] = 1.0
    center = np.array([0.0, 0.0, 0.0])
    radius = 1.0

    t = intersect_buggy(origins, dirs, center, radius)
    print("hit distances (far root):", np.round(t, 3))
