"""
gen_demo — helper library for Module 05 (Particle-physics generators on the GPU).

Everything here is designed for a **Linux node with CVMFS + Key4hep** — nothing
is installed locally. MadGraph5_aMC@NLO, PYTHIA 8, compilers and (on GPU nodes)
`nvcc` all come from CVMFS after:

    source /cvmfs/sw.hsf.org/key4hep/setup.sh

The notebook imports this module and calls its functions, keeping the notebook
cells short (the "program files" are factored out here, as in modules 07 and 11).

Public helpers:
    verify_toolchain()               -> dict of discovered tools
    generate_cpu(process, ...)       -> real LO cross section on CPU via mg5_aMC
    matrix_element_gpu_benchmark(...)-> CUDACPP matrix element, CPU vs GPU throughput
    toy_matrix_element(...)          -> illustrative NumPy-vs-CuPy matrix element

Heavy operations (real generation, CUDACPP build) are explicit and opt-in; every
function degrades gracefully with a clear message when a tool is missing.
"""
from __future__ import annotations

import glob
import math
import os
import pathlib
import re
import subprocess
import tempfile
import time

# Key4hep environment on CVMFS (provides mg5_aMC, pythia8, compilers, ...).
KEY4HEP_SETUP = "/cvmfs/sw.hsf.org/key4hep/setup.sh"


# --------------------------------------------------------------------------- #
# Shell helpers (always run through a Key4hep-sourced bash)
# --------------------------------------------------------------------------- #
def _sh(cmd: str, cwd=None, timeout=None) -> subprocess.CompletedProcess:
    """Run a bash command with Key4hep (CVMFS) sourced first, so mg5_aMC, the
    compilers, and any built binaries all see the CVMFS environment — regardless
    of how Jupyter itself was launched."""
    full = f'source "{KEY4HEP_SETUP}" >/dev/null 2>&1 || true; {cmd}'
    try:
        return subprocess.run(
            ["bash", "-lc", full], cwd=cwd,
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(["bash"], 127, "", "bash not found")
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(["bash"], 124, exc.stdout or "", "timed out")


def which_cvmfs(tool: str):
    """Return the CVMFS path of `tool` (via a Key4hep-sourced shell) or None."""
    return _sh(f"command -v {tool}").stdout.strip() or None


def verify_toolchain() -> dict:
    """Print which generator / GPU tools are visible via CVMFS / Key4hep."""
    tools = ["mg5_aMC", "pythia8-config", "nvcc", "make", "g++", "python3"]
    print("Toolchain discovered via CVMFS / Key4hep:")
    found = {}
    for t in tools:
        p = which_cvmfs(t)
        found[t] = p
        print(f"  {t:14s}: {p or 'NOT FOUND'}")
    if not found["mg5_aMC"]:
        print("\n  mg5_aMC missing -> run:  source /cvmfs/sw.hsf.org/key4hep/setup.sh")
    if not found["nvcc"]:
        print("  nvcc missing -> the GPU (CUDA) backend is unavailable on this node")
    return found


# --------------------------------------------------------------------------- #
# 1) Real CPU generation with MadGraph5_aMC@NLO
# --------------------------------------------------------------------------- #
_XSEC_RE = re.compile(r"Cross-section\s*:\s*([0-9.eE+\-]+)\s*\+-\s*([0-9.eE+\-]+)\s*pb")


def generate_cpu(process: str = "g g > t t~", ecm: float = 13000.0,
                 nevents: int = 2000, workdir=None):
    """Run the real MadGraph5_aMC@NLO on the CPU (from CVMFS) to compute the LO
    cross section for `process` and generate parton-level events.

    Returns a dict {process, cross_section_pb, error_pb, nevents, time_s, outdir}
    or None if mg5_aMC is unavailable or generation failed. This launches the
    standard MadEvent workflow, so it compiles Fortran and runs on the CPU — it
    can take from tens of seconds to a few minutes.
    """
    if not which_cvmfs("mg5_aMC"):
        print("mg5_aMC not found. First: source /cvmfs/sw.hsf.org/key4hep/setup.sh")
        return None

    work = pathlib.Path(workdir or tempfile.mkdtemp(prefix="mg5cpu_"))
    work.mkdir(parents=True, exist_ok=True)
    outdir = "proc_cpu"
    (work / "cpu.mg").write_text(
        f"generate {process}\n"
        f"output {outdir}\n"
        f"launch\n"
        f"set nevents {int(nevents)}\n"
        f"set ebeam1 {ecm / 2}\n"
        f"set ebeam2 {ecm / 2}\n"
        f"set use_syst False\n"
        f"0\n"
    )
    print(f"Generating '{process}' on CPU with mg5_aMC "
          f"(nevents={nevents}, sqrt(s)={ecm/1000:.1f} TeV) ...")
    t0 = time.perf_counter()
    r = _sh("mg5_aMC cpu.mg", cwd=work, timeout=3600)
    dt = time.perf_counter() - t0

    xsec = _parse_cross_section(r.stdout, work / outdir)
    if xsec is None:
        print("Could not parse a cross section; tail of mg5_aMC output:\n",
              (r.stdout + r.stderr)[-1800:])
        return None

    print(f"- LO cross section : {xsec[0]:.4g} +/- {xsec[1]:.2g} pb")
    print(f"- Events generated : {nevents}")
    print(f"- CPU wall time    : {dt:.1f} s")
    return {
        "process": process, "cross_section_pb": xsec[0], "error_pb": xsec[1],
        "nevents": nevents, "time_s": dt, "outdir": str(work / outdir),
    }


def _parse_cross_section(stdout: str, outdir: pathlib.Path):
    """Best-effort extraction of (value, error) in pb from mg5_aMC output."""
    matches = _XSEC_RE.findall(stdout or "")
    if matches:
        v, e = matches[-1]
        try:
            return float(v), float(e)
        except ValueError:
            pass
    # Fallback: scan the run banner(s) written under the output directory.
    for banner in glob.glob(str(outdir / "Events" / "**" / "*banner*.txt"), recursive=True):
        try:
            txt = pathlib.Path(banner).read_text(errors="ignore")
        except OSError:
            continue
        m = _XSEC_RE.findall(txt)
        if m:
            try:
                return float(m[-1][0]), float(m[-1][1])
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------- #
# 2) Real GPU backend: MadGraph CUDACPP matrix element (CPU vs GPU throughput)
# --------------------------------------------------------------------------- #
_THR_RE = re.compile(r"EvtsPerSec\[MatrixElems[^\]]*\][^=]*=\s*\(?\s*([0-9.eE+]+)")


def _throughput(stdout: str):
    m = _THR_RE.search(stdout or "")
    return float(m.group(1)) if m else None


def matrix_element_gpu_benchmark(process: str = "g g > t t~",
                                 run: bool = False, workdir=None):
    """Generate the MadGraph **CUDACPP** standalone for `process` with the CVMFS
    mg5_aMC, build the CPU and (if `nvcc`) CUDA backends, and benchmark the
    matrix-element throughput on CPU vs GPU. Nothing is cloned or installed.

    Detection only by default; pass run=True to generate + build + benchmark.
    Returns a dict of results when run=True, else None.
    """
    have = {t: which_cvmfs(t) for t in ("mg5_aMC", "make", "g++", "nvcc")}
    print("From CVMFS / Key4hep:")
    for t, p in have.items():
        print(f"  {t:8s}: {p or 'NOT FOUND'}")

    if not have["mg5_aMC"]:
        print("\nmg5_aMC not found -> source /cvmfs/sw.hsf.org/key4hep/setup.sh")
        return None
    if not have["nvcc"]:
        print("\nNote: no nvcc -> only the CPU backend builds (no GPU comparison).")
    if not run:
        print("\nDetection only. To generate + build + benchmark, call with run=True.")
        return None

    work = pathlib.Path(workdir or tempfile.mkdtemp(prefix="cudacpp_"))
    work.mkdir(parents=True, exist_ok=True)
    outdir = "proc_cudacpp"
    (work / "proc.mg").write_text(
        f"generate {process}\noutput standalone_cudacpp {outdir}\n"
    )
    print(f"\nGenerating standalone CUDACPP code for '{process}' ...")
    r = _sh("mg5_aMC proc.mg", cwd=work, timeout=1800)
    gen = work / outdir
    if r.returncode != 0 or not gen.is_dir():
        print("  generation failed; tail of output:\n", (r.stdout + r.stderr)[-1800:])
        return None

    subs = sorted((gen / "SubProcesses").glob("P1_*"))
    if not subs:
        print("  generated output but no P1_* SubProcess directory found.")
        return None
    pdir = subs[0]
    print("Process dir:", pdir)

    print("\nBuilding (make -j: CPU backend, plus CUDA if nvcc is present) ...")
    rb = _sh("make -j", cwd=pdir, timeout=1800)
    if rb.returncode != 0:
        print("  build did not fully succeed; tail of stderr:\n", rb.stderr[-1200:])

    exes = sorted(set(
        glob.glob(str(pdir / "**" / "check*.exe"), recursive=True)
        + glob.glob(str(pdir / "**" / "gcheck*.exe"), recursive=True)
    ))
    if not exes:
        print("\nNo check executables were produced; inspect the build output above.")
        return None

    print("\nMatrix-element throughput (higher is better):")
    results, cpu_best, gpu_best = {}, 0.0, 0.0
    for exe in exes:
        is_gpu = ("cuda" in exe.lower()) or ("gcheck" in os.path.basename(exe).lower())
        label = "GPU (CUDA)" if is_gpu else "CPU"
        rr = _sh(f'"{exe}" -p 2048 256 12', timeout=600)
        thr = _throughput(rr.stdout)
        results[os.path.basename(exe)] = (label, thr)
        print(f"  {label:11s} {os.path.basename(exe):24s} "
              + (f"{thr:.3e} ME/s" if thr else "(throughput not parsed)"))
        if thr and is_gpu:
            gpu_best = max(gpu_best, thr)
        elif thr:
            cpu_best = max(cpu_best, thr)
    if cpu_best and gpu_best:
        print(f"\n  GPU vs best CPU throughput: {gpu_best / cpu_best:.1f}x")
    return {"process": process, "results": results,
            "cpu_best": cpu_best, "gpu_best": gpu_best}


# --------------------------------------------------------------------------- #
# 3) Intuition builder: the same matrix element on CPU (NumPy) vs GPU (CuPy)
# --------------------------------------------------------------------------- #
def toy_matrix_element(nevents: int = 20_000_000, beam_energy: float = 500.0):
    """Illustrative, dependency-light comparison: evaluate the leading-order QED
    angular matrix element for e+e- -> mu+mu- (proportional to 1 + cos^2(theta))
    over many phase-space points, on the CPU with NumPy and — if available — on
    the GPU with CuPy. This mirrors *why* real matrix-element evaluation is a good
    GPU workload: the same arithmetic over millions of independent points.

    Returns a dict with the CPU/GPU timings, cross-section estimate, and speedup.
    """
    import numpy as np

    alpha = 1.0 / 137.0
    s = (2.0 * beam_energy) ** 2
    gev2_to_pb = 0.389379338e9
    sigma_analytic = (4.0 * math.pi * alpha ** 2 / (3.0 * s)) * gev2_to_pb

    def _cpu():
        rng = np.random.default_rng(42)
        t0 = time.perf_counter()
        cos_t = rng.uniform(-1.0, 1.0, nevents)
        dsig = (alpha ** 2 / (4.0 * s)) * (1.0 + cos_t ** 2)
        sigma = 4.0 * np.pi * float(np.mean(dsig)) * gev2_to_pb
        return sigma, time.perf_counter() - t0

    sigma_cpu, t_cpu = _cpu()
    out = {
        "nevents": nevents,
        "sigma_analytic_pb": sigma_analytic,
        "sigma_cpu_pb": sigma_cpu,
        "cpu_time_s": t_cpu,
        "sigma_gpu_pb": None,
        "gpu_time_s": None,
        "speedup": None,
        "backend": "CPU (NumPy) only",
    }
    try:
        import cupy as cp  # noqa: F401
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        cos_t = cp.random.uniform(-1.0, 1.0, nevents, dtype=cp.float32)
        dsig = (alpha ** 2 / (4.0 * s)) * (1.0 + cos_t ** 2)
        sigma_gpu = 4.0 * cp.pi * cp.mean(dsig) * gev2_to_pb
        end.record()
        end.synchronize()
        out["gpu_time_s"] = cp.cuda.get_elapsed_time(start, end) / 1e3
        out["sigma_gpu_pb"] = float(sigma_gpu)
        out["speedup"] = t_cpu / out["gpu_time_s"] if out["gpu_time_s"] else None
        out["backend"] = f"CPU (NumPy) + GPU (CuPy, {cp.cuda.Device().name})"
    except Exception as exc:  # cupy missing or no GPU
        out["gpu_note"] = f"GPU path skipped ({str(exc)[:80]})"

    print(f"Backend: {out['backend']}")
    print(f"Events : {nevents:,}")
    print(f"Analytic sigma : {sigma_analytic:.5f} pb")
    print(f"CPU  : {sigma_cpu:.5f} pb  in {t_cpu*1e3:.1f} ms")
    if out["sigma_gpu_pb"] is not None:
        print(f"GPU  : {out['sigma_gpu_pb']:.5f} pb  in {out['gpu_time_s']*1e3:.1f} ms")
        print(f"Speedup (CPU/GPU): {out['speedup']:.1f}x")
    else:
        print(out.get("gpu_note", "GPU path unavailable"))
    return out
