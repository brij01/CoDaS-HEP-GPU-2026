# Module 05 — factored helpers (`gen-demo/`)

Runnable code for the generator notebook is factored out here (as in modules 07
and 11) so the notebook cells stay short.

- [`gen_demo.py`](gen_demo.py) — everything the notebook calls:
  - `verify_toolchain()` — report the generator/GPU tools visible via CVMFS/Key4hep.
  - `generate_cpu(process, ...)` — real LO cross section on the **CPU** via `mg5_aMC`.
  - `matrix_element_gpu_benchmark(process, run=...)` — MadGraph **CUDACPP** matrix
    element, **CPU vs GPU** throughput (generate → build → benchmark).
  - `toy_matrix_element(...)` — illustrative **NumPy vs CuPy** matrix-element kernel.

Everything targets a **Linux node with CVMFS + Key4hep** — nothing is installed
locally. Source Key4hep first:

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh
```

The helpers also source Key4hep in a subshell, so they work whether or not the
Jupyter kernel inherited the environment. Heavy operations (real generation,
CUDACPP build) are explicit/opt-in and degrade gracefully when a tool is missing.
