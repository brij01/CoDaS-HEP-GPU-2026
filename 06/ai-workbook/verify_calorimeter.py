"""VERIFICATION HARNESS -- Phase 4 for Module 06.

This is YOUR test harness. It ships with a CPU reference (weighted np.bincount)
and two gates: total-energy conservation and a per-cell comparison. Implement the
function under test and run.

    python verify_calorimeter.py
"""

import numpy as np


def deposit_reference(cell_ids, edep, ncells):
    """Source of truth: exact per-cell sum via weighted bincount."""
    return np.bincount(cell_ids, weights=edep, minlength=ncells).astype(np.float64)


def deposit_under_test(cell_ids, edep, ncells):
    """TODO (student): implement (or paste the AI's) deposition.

    A correct fix: np.add.at(cells, cell_ids, edep)   (GPU: cuda.atomic.add).
    The default below is the buggy version and will FAIL.
    """
    cells = np.zeros(ncells, dtype=np.float64)
    cells[cell_ids] += edep          # buggy default
    return cells


def main():
    rng = np.random.default_rng(7)
    nhits, ncells = 500_000, 2048
    cell_ids = rng.integers(0, ncells, size=nhits)
    edep = rng.uniform(0.1, 5.0, size=nhits)

    ref = deposit_reference(cell_ids, edep, ncells)
    got = deposit_under_test(cell_ids, edep, ncells)

    energy_ok = np.isclose(got.sum(), edep.sum(), rtol=1e-9)
    cells_ok = np.allclose(got, ref, rtol=1e-9)
    if energy_ok and cells_ok:
        print(f"PASS: energy conserved ({got.sum():.3f}) and all {ncells} cells match")
    else:
        print(f"FAIL: energy in map {got.sum():.3f} vs deposited {edep.sum():.3f}; "
              f"cells match = {cells_ok}")


if __name__ == "__main__":
    main()
