"""SOLUTION -- correct calorimeter deposition (accumulating scatter-add) plus the
GPU (Numba CUDA) atomic analog.

The bug was `cells[cell_id] += edep` with repeated ids. Fix: np.add.at (CPU) or
cuda.atomic.add (GPU).

    python calorimeter_solution.py
"""

import numpy as np


def deposit_reference(cell_ids, edep, ncells):
    return np.bincount(cell_ids, weights=edep, minlength=ncells).astype(np.float64)


def deposit_correct(cell_ids, edep, ncells):
    cells = np.zeros(ncells, dtype=np.float64)
    np.add.at(cells, cell_ids, edep)         # unbuffered accumulating scatter-add
    return cells


def deposit_gpu(cell_ids, edep, ncells):
    """The same fix on the GPU: every step atomically adds into its cell."""
    from numba import cuda

    @cuda.jit
    def _kernel(ids, e, cells):
        i = cuda.grid(1)
        if i < ids.size:
            cuda.atomic.add(cells, ids[i], e[i])   # atomic: energy conserved

    d_ids = cuda.to_device(cell_ids.astype(np.int32))
    d_e = cuda.to_device(edep.astype(np.float64))
    d_cells = cuda.to_device(np.zeros(ncells, dtype=np.float64))
    threads = 256
    blocks = (cell_ids.size + threads - 1) // threads
    _kernel[blocks, threads](d_ids, d_e, d_cells)
    return d_cells.copy_to_host()


def main():
    rng = np.random.default_rng(7)
    nhits, ncells = 500_000, 2048
    cell_ids = rng.integers(0, ncells, size=nhits)
    edep = rng.uniform(0.1, 5.0, size=nhits)

    ref = deposit_reference(cell_ids, edep, ncells)
    cpu = deposit_correct(cell_ids, edep, ncells)
    print("CPU fix:", "PASS" if np.allclose(cpu, ref) and
          np.isclose(cpu.sum(), edep.sum()) else "FAIL")

    try:
        gpu = deposit_gpu(cell_ids, edep, ncells)
        print("GPU (atomic):", "PASS" if np.allclose(gpu, ref) else "FAIL")
    except Exception as exc:  # noqa: BLE001 -- no GPU / numba not installed
        print(f"GPU check skipped (no CUDA device or numba): {exc}")


if __name__ == "__main__":
    main()
