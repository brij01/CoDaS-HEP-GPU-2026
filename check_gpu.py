#!/usr/bin/env python3
"""GPU environment check for the CoDaS-HEP 2026 GPU course.

Run it directly from a terminal::

    python check_gpu.py

It is also invoked by the "Prerequisites" cell at the top of each course
notebook. It reports OK / MISSING for the NVIDIA command-line tools and the
Python GPU packages the modules use, and prints how to fix anything missing.

Required (the course cannot run without these):
    - nvidia-smi   GPU driver + runtime
    - nvcc         CUDA compiler (needed for every .cu example)
    - numpy        CPU reference implementations in the Python workbooks

Optional (only some paths need these; the workbooks skip them gracefully):
    - nsys         Nsight Systems profiler   (Phase-5 profiling, Module 03)
    - ncu          Nsight Compute profiler   (kernel-level profiling)
    - numba        GPU path in Modules 04, 06, 10
    - cupy         GPU path in Modules 07, 10
"""
from __future__ import annotations

import shutil
import subprocess
import sys

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if _supports_color() else text


def _run(cmd) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - report any failure verbatim
        return f"error: {exc}"


def _first_line(text: str) -> str:
    return text.splitlines()[0].strip() if text else ""


def _status(ok: bool, required: bool) -> str:
    label = "OK" if ok else ("MISSING" if required else "optional")
    color = GREEN if ok else (RED if required else YELLOW)
    return _c(f"{label:>8}", color)


def check_cli_tools():
    """Return (all_required_ok, rows) for the NVIDIA command-line tools."""
    tools = [
        # name, description, required, version-probe command
        ("nvidia-smi", "GPU driver + runtime", True,
         ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
          "--format=csv,noheader"]),
        ("nvcc", "CUDA compiler (.cu files)", True, ["nvcc", "--version"]),
        ("nsys", "Nsight Systems profiler", False, ["nsys", "--version"]),
        ("ncu", "Nsight Compute profiler", False, ["ncu", "--version"]),
    ]
    rows = []
    all_required_ok = True
    for name, desc, required, probe in tools:
        path = shutil.which(name)
        ok = path is not None
        detail = ""
        if ok:
            detail = _first_line(_run(probe))
        elif required:
            all_required_ok = False
        rows.append((_status(ok, required), name, desc, detail))
    return all_required_ok, rows


def check_python_packages():
    """Return (all_required_ok, rows) for the Python GPU packages."""
    rows = []
    all_required_ok = True

    # numpy (required)
    try:
        import numpy  # noqa: WPS433 - runtime probe
        rows.append((_status(True, True), "numpy", "CPU reference implementations",
                     numpy.__version__))
    except Exception:  # noqa: BLE001
        rows.append((_status(False, True), "numpy", "CPU reference implementations", ""))
        all_required_ok = False

    # numba CUDA (optional)
    try:
        from numba import cuda  # noqa: WPS433
        avail = bool(cuda.is_available())
        detail = "GPU visible" if avail else "installed but GPU NOT visible"
        rows.append((_status(avail, False), "numba", "GPU path (Modules 04, 06, 10)",
                     detail))
    except Exception as exc:  # noqa: BLE001
        rows.append((_status(False, False), "numba", "GPU path (Modules 04, 06, 10)",
                     f"not installed ({type(exc).__name__})"))

    # cupy (optional)
    try:
        import cupy  # noqa: WPS433
        ndev = cupy.cuda.runtime.getDeviceCount()
        rows.append((_status(ndev > 0, False), "cupy", "GPU path (Modules 07, 10)",
                     f"{ndev} device(s), v{cupy.__version__}"))
    except Exception as exc:  # noqa: BLE001
        rows.append((_status(False, False), "cupy", "GPU path (Modules 07, 10)",
                     f"not installed ({type(exc).__name__})"))

    return all_required_ok, rows


def _print_table(title, rows):
    print(title)
    for status, name, desc, detail in rows:
        line = f"  [{status}] {name:<11} {desc}"
        if detail:
            line += f"  -  {detail}"
        print(line)
    print()


def main() -> int:
    print("=" * 68)
    print("CoDaS-HEP 2026 - GPU environment check")
    print("=" * 68)

    cli_ok, cli_rows = check_cli_tools()
    _print_table("NVIDIA command-line tools:", cli_rows)

    py_ok, py_rows = check_python_packages()
    _print_table("Python GPU packages:", py_rows)

    required_ok = cli_ok and py_ok
    if required_ok:
        print(_c("All REQUIRED tools are present — you can run the course.", GREEN))
        print("Any 'optional' items above only affect specific bonus paths;")
        print("the workbooks skip them gracefully when absent.")
    else:
        print(_c("Some REQUIRED tools are MISSING — see fixes below.", RED))
        print()
        print("Fixes:")
        print("  - nvidia-smi : install the NVIDIA GPU driver for your card.")
        print("  - nvcc       : install the CUDA Toolkit (or NVIDIA HPC SDK) and put")
        print("                 its bin/ on PATH.")
        print("  - numpy      : pip install numpy")
        print()
        print("Optional GPU packages:")
        print("  - numba : pip install -U numba-cuda   (set CUDA_HOME if GPU not visible)")
        print("  - cupy  : pip install cupy-cuda13x    (match your CUDA major version)")

    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
