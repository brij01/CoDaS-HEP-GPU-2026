"""
adept_demo — helper for the AdePT (GPU EM transport) part of Module 06.

AdePT (https://github.com/apt-sim/AdePT) is a lightweight **Geant4 plugin that
offloads electromagnetic transport of e-/e+/gamma to the GPU** (built on
G4HepEm + VecGeom). It is the sibling R&D project to Celeritas.

Everything runs from CVMFS — source the `devAdePT` LCG view:

    source /cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-el9-gcc13-opt/setup.sh

That view provides the AdePT environment (and, on the school nodes, the built
`example1` standalone Geant4+AdePT application). This module discovers and runs
it; if `example1` is not on `PATH`, it prints the one-time build commands.

Every function sources the view in a subshell, so it works regardless of how
Jupyter was launched, and degrades gracefully when a tool is missing.
"""
from __future__ import annotations

import pathlib
import subprocess
import tempfile
import time

# AdePT environment on CVMFS (LCG devAdePT view).
ADEPT_SETUP = "/cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-el9-gcc13-opt/setup.sh"

_BUILD_HINT = """\
example1 was not found on the devAdePT view. Build it once (needs a GPU node):

  source /cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-el9-gcc13-opt/setup.sh
  git clone https://github.com/apt-sim/AdePT.git
  cd AdePT
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  cmake -S . -B adept-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CC
  cmake --build adept-build -- -j6
  ./adept-build/BuildProducts/bin/example1 -m adept-build/example1.mac
"""


def _sh(cmd: str, setup: str = ADEPT_SETUP, cwd=None, timeout=None):
    """Run a bash command with the given CVMFS environment sourced first."""
    full = f'source "{setup}" >/dev/null 2>&1 || true; {cmd}'
    try:
        return subprocess.run(["bash", "-lc", full], cwd=cwd,
                              capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return subprocess.CompletedProcess(["bash"], 127, "", "bash not found")
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(["bash"], 124, exc.stdout or "", "timed out")


def _which(tool: str):
    return _sh(f"command -v {tool}").stdout.strip() or None


def verify_adept() -> dict:
    """Report the AdePT toolchain visible from the devAdePT CVMFS view."""
    print("From the devAdePT CVMFS view:")
    ex = find_example1()
    print(f"  example1      : {ex or 'NOT FOUND (build it once — see run_example1 output)'}")
    print(f"  geant4-config : {_which('geant4-config') or 'NOT FOUND'}")
    print(f"  nvcc          : {_which('nvcc') or 'NOT FOUND'}")
    gpu = _sh("command -v nvidia-smi >/dev/null 2>&1 && "
              "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo 'no GPU'")
    print(f"  GPU           : {gpu.stdout.strip() or 'unknown'}")
    return {"example1": ex}


def find_example1():
    """Locate the example1 executable (on PATH via the view, or a local build)."""
    ex = _which("example1")
    if ex:
        return ex
    for guess in ("adept-build/BuildProducts/bin/example1",
                  str(pathlib.Path.home() / "adept-build/BuildProducts/bin/example1"),
                  str(pathlib.Path.home() / "AdePT/adept-build/BuildProducts/bin/example1")):
        if pathlib.Path(guess).is_file():
            return str(pathlib.Path(guess).resolve())
    return None


def _resolve_macro(example1_path: str, macro: str):
    """Find the macro file: as given, next to the executable, or via `find`."""
    p = pathlib.Path(macro)
    if p.is_file():
        return str(p.resolve())
    exdir = pathlib.Path(example1_path).resolve().parent
    for base in (exdir, exdir.parent, exdir.parent.parent):
        cand = base / macro
        if cand.is_file():
            return str(cand)
    r = _sh(f'find "{exdir.parent.parent}" -name "{macro}" 2>/dev/null | head -n1')
    return r.stdout.strip() or None


def run_example1(macro: str = "example1.mac", run: bool = False,
                 label: str = "AdePT (GPU EM transport)", timeout: int = 1800):
    """Run the AdePT standalone Geant4 example (`example1`) with a macro.

    `example1` offloads electromagnetic transport to the GPU, so this is the GPU
    example. Detection only unless run=True. Returns {label, time_s, stdout_tail}
    or None if example1 / the macro could not be found.

    For a CPU baseline of the same idea, use the Celeritas section's
    CELER_DISABLE_DEVICE run, or a standard-Geant4 build without the AdePT physics
    constructor (see the notebook and `example1 -h`).
    """
    ex = find_example1()
    print(f"example1: {ex or 'NOT FOUND'}")
    if ex is None:
        print("\n" + _BUILD_HINT)
        return None
    if not run:
        print("\nDetection only. To run it on a GPU node, call run_example1(run=True).")
        return None

    mac = _resolve_macro(ex, macro)
    if mac is None:
        print(f"Could not find the macro '{macro}'. Pass an explicit path, e.g. "
              f"run_example1(macro='adept-build/example1.mac', run=True).")
        return None

    work = pathlib.Path(tempfile.mkdtemp(prefix="adept_"))
    print(f"\nRunning {label}:\n  {ex} -m {mac}")
    t0 = time.perf_counter()
    r = _sh(f'"{ex}" -m "{mac}"', cwd=work, timeout=timeout)
    dt = time.perf_counter() - t0
    tail = (r.stdout + r.stderr)[-2000:]
    print(f"- wall time: {dt:.1f} s")
    print("- output (tail):\n", tail)
    return {"label": label, "time_s": dt, "stdout_tail": tail}
