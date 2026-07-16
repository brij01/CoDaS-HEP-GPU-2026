# Module 06 — AdePT helpers (`adept-demo/`)

Factored code for the **AdePT** part of the GPU-simulation notebook (as in
modules 05, 07, 11), so the notebook cells stay short.

[AdePT](https://github.com/apt-sim/AdePT) — *Accelerated demonstrator of
electromagnetic Particle Transport* — is a lightweight **Geant4 plugin that
offloads EM transport of e⁻/e⁺/γ to the GPU** (built on G4HepEm + VecGeom). It is
the sibling R&D project to Celeritas.

- [`adept_demo.py`](adept_demo.py):
  - `verify_adept()` — report the AdePT toolchain from the CVMFS view.
  - `find_example1()` — locate the `example1` standalone Geant4+AdePT app.
  - `run_example1(macro, run=...)` — run `example1` on the GPU (detection-only by
    default; prints the one-time build commands if `example1` isn't on `PATH`).

Everything comes from CVMFS — source the `devAdePT` LCG view first:

```bash
source /cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-el9-gcc13-opt/setup.sh
```

The helpers source the view in a subshell, so they work whether or not the
Jupyter kernel inherited it. Running the example needs a **GPU node**.
