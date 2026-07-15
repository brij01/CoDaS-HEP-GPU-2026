"""ADVERSARIAL EXAMPLE -- this calorimeter hit-map accumulation "an AI wrote for
you" is INTENTIONALLY BUGGY. It runs and produces a plausible energy map, but
total energy is NOT conserved: `cells[cell_id] += edep` with repeated cell ids
does not accumulate (NumPy applies each duplicate index once). This is the CPU
mirror of a GPU energy deposition without cuda.atomic.add.

Run it:
    python adversarial_calorimeter_buggy.py
"""

import numpy as np


def deposit_buggy(cell_ids, edep, ncells):
    cells = np.zeros(ncells, dtype=np.float64)
    cells[cell_ids] += edep          # BUG: repeated cell_ids are NOT summed
    return cells


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    nhits = 500_000
    ncells = 2048
    # Many hits fall in the same cells (a real shower), triggering the bug.
    cell_ids = rng.integers(0, ncells, size=nhits)
    edep = rng.uniform(0.1, 5.0, size=nhits)

    cells = deposit_buggy(cell_ids, edep, ncells)
    print(f"total energy deposited : {edep.sum():.3f}")
    print(f"total energy in map    : {cells.sum():.3f}")
    print(f"energy lost            : {edep.sum() - cells.sum():.3f}")
