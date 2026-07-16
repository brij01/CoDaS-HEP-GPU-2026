---
jupyter:
  jupytext:
    formats: md,ipynb
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---


# GPU-Accelerated Geant4: Celeritas and AdePT with TileCal Geometry

> Checked against upstream documentation in May 2026.
>
> Official Celeritas docs: https://celeritas-project.github.io/celeritas/user/index.html
> Celeritas quick start: https://celeritas-project.github.io/celeritas/user/index.html#quick-start-guide
> Geant4 integration examples: https://github.com/celeritas-project/celeritas/tree/main/example/geant4
> AdePT project: https://github.com/apt-sim/AdePT
> DD4hep project page: https://dd4hep.web.cern.ch/dd4hep/
> Geant4 documentation: https://geant4.web.cern.ch/documentation
> TileCal geometry and macro source: https://github.com/celeritas-project/atlas-tilecal-integration
>
> Note: the TileCal repository is still useful for the GDML geometry and macro cards, but its build and integration instructions predate the current Celeritas Geant4 API. In this notebook we use that repository only as a source of geometry and macro files.

Detector simulation is one of the largest CPU consumers in HEP, and the hot path is **electromagnetic transport** — the e⁻/e⁺/γ showers inside calorimeters. Two CERN R&D projects offload exactly that to the **GPU** while Geant4 keeps the rest of the event on the CPU:

- **[Celeritas](https://github.com/celeritas-project/celeritas)** — GPU transport library with a Geant4 tracking-manager offload (Apache-2.0).
- **[AdePT](https://github.com/apt-sim/AdePT)** — lightweight Geant4 plugin that offloads EM transport via G4HepEm + VecGeom (Apache-2.0).

This notebook has two goals — **run simulation on the GPU** (same physics on CPU vs GPU, measure the speed-up) and **show where you can contribute** to these active projects. Everything runs from **CVMFS**, so nothing is installed locally:

1. Set up a prebuilt Geant4 + Celeritas toolchain from CVMFS (Key4hep) — no Spack, no build.
2. Download the TileCal GDML geometry and macro files.
3. Build a small Geant4 application using the current Celeritas tracking-manager integration path.
4. **Celeritas:** compare CPU-only and GPU-enabled runs and inspect the diagnostics.
5. **AdePT:** run the standalone Geant4 + AdePT `example1` (GPU EM transport) from the CVMFS `devAdePT` view.
6. **Contribute:** the GPU-simulation landscape and good first issues.

The example below uses the TileCal geometry distributed in the DD4hep/Celeritas ecosystem, but the executable itself is a plain Geant4 application that loads GDML directly.

> This Markdown mirrors `gpu_dd4hep_tilecal.ipynb`; the notebook is the canonical, runnable version.

---

## 0 Environment setup — Key4hep from CVMFS

This notebook targets a **Linux node with an NVIDIA GPU** and a mounted **CVMFS**. Instead of building Geant4/Celeritas from source, it uses the prebuilt **Key4hep** stack from CVMFS, which ships Geant4, DD4hep, and Celeritas ready to use — so there is **no `spack install` and no long compile**.

The whole setup is a single `source`:

```bash
# %%bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh
set -e
geant4-config --version && echo "geant4-config: OK"
cmake --version | head -n1
command -v nvidia-smi >/dev/null 2>&1 \
  && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
  || echo "nvidia-smi: not found (GPU runs will fall back to CPU)"
```

**Recommended:** source Key4hep in your shell *before* launching Jupyter / VS Code so the kernel inherits it. The notebook does not rely on that — every `%%bash` build/run cell re-sources Key4hep, and the Python run helper does too. Add `-r <YYYY-MM-DD>` to pin a release.

---

## 1 What the Key4hep stack provides

Sourcing `/cvmfs/sw.hsf.org/key4hep/setup.sh` puts a consistent, prebuilt HEP toolchain on your `PATH` and `CMAKE_PREFIX_PATH`:

- **Geant4** (11.x) — with the tracking-manager offload interface Celeritas needs.
- **DD4hep** — detector description (the minimal example reads GDML directly, but DD4hep matches the broader workflow).
- **Celeritas** — the GPU transport library, built with CUDA in the CVMFS release.
- **CMake** and a C++17 compiler.

Because the stack is prebuilt on CVMFS there is **no `spack install` and no long compile**. If the release you sourced lacks Celeritas, pin a newer one with `-r <YYYY-MM-DD>`.

GPU note: whether a run uses the GPU is decided at **runtime**, not build time. The comparison cells toggle `CELER_DISABLE_DEVICE` to force CPU-only vs GPU execution of the *same* binary; if no NVIDIA GPU is visible, Celeritas falls back to CPU. Each `%%bash` cell re-sources Key4hep, and the Python run helper wraps the executable in a Key4hep-sourced shell, so the notebook works whether or not you launched Jupyter from a Key4hep shell.

---

## 2 Download the TileCal geometry and macro files

The upstream TileCal repository still provides useful input files. The macro files we use here are `TBrun.mac` and `TBrun_all.mac`.

```python
import pathlib
import urllib.request

tile_repo = "https://raw.githubusercontent.com/celeritas-project/atlas-tilecal-integration/main/"
for filename in (
    "TileTB_2B1EB_nobeamline.gdml",
    "TBrun.mac",
    "TBrun_all.mac",
):
    path = pathlib.Path(filename)
    if not path.exists():
        urllib.request.urlretrieve(tile_repo + filename, path)

print("Downloaded:")
for path in sorted(pathlib.Path(".").glob("TileTB_*")):
    print(" -", path)
for path in sorted(pathlib.Path(".").glob("TBrun*.mac")):
    print(" -", path)
```

---

## 3 Write a minimal Geant4 plus Celeritas application

The current upstream integration pattern is:

1. Include `CeleritasG4.hh`.
2. Use `TrackingManagerIntegration::Instance()`.
3. Register `TrackingManagerConstructor` on your Geant4 physics list.
4. Call `BeginOfRunAction` and `EndOfRunAction` from a Geant4 run action.
5. Link with `Celeritas::G4` and use `celeritas_target_link_libraries(...)` in CMake.

The code below loads the TileCal GDML directly through Geant4's GDML parser and executes a Geant4 macro card.

```cpp
// %%writefile tile_gpu.cc
#include <memory>
#include <string>

#include <FTFP_BERT.hh>
#include <G4GDMLParser.hh>
#include <G4ParticleGun.hh>
#include <G4ParticleTable.hh>
#include <G4RunManagerFactory.hh>
#include <G4SystemOfUnits.hh>
#include <G4ThreeVector.hh>
#include <G4UImanager.hh>
#include <G4UserRunAction.hh>
#include <G4VUserActionInitialization.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4VUserPrimaryGeneratorAction.hh>

#include <CeleritasG4.hh>

using TMI = celeritas::TrackingManagerIntegration;

namespace
{
class DetectorConstruction final : public G4VUserDetectorConstruction
{
  public:
    G4VPhysicalVolume* Construct() final
    {
        parser_.Read("TileTB_2B1EB_nobeamline.gdml", false);
        return parser_.GetWorldVolume();
    }

  private:
    G4GDMLParser parser_;
};

class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction()
        : gun_(1)
    {
        auto* particle
            = G4ParticleTable::GetParticleTable()->FindParticle("e-");
        gun_.SetParticleDefinition(particle);
        gun_.SetParticleEnergy(18 * GeV);
        gun_.SetParticlePosition(G4ThreeVector{0, 0, 0});
        gun_.SetParticleMomentumDirection(G4ThreeVector{1, 0, 0});
    }

    void GeneratePrimaries(G4Event* event) final
    {
        gun_.GeneratePrimaryVertex(event);
    }

  private:
    G4ParticleGun gun_;
};

class RunAction final : public G4UserRunAction
{
  public:
    void BeginOfRunAction(G4Run const* run) final
    {
        TMI::Instance().BeginOfRunAction(run);
    }

    void EndOfRunAction(G4Run const* run) final
    {
        TMI::Instance().EndOfRunAction(run);
    }
};

class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    void BuildForMaster() const final
    {
        this->SetUserAction(new RunAction{});
    }

    void Build() const final
    {
        this->SetUserAction(new PrimaryGeneratorAction{});
        this->SetUserAction(new RunAction{});
    }
};

celeritas::SetupOptions MakeOptions()
{
    celeritas::SetupOptions options;
    options.max_num_tracks = 4096;
    options.initializer_capacity = 4096 * 128;
    options.ignore_processes = {"CoulombScat"};
    options.output_file = "tile_gpu.out.json";
    return options;
}
}  // namespace

int main(int argc, char* argv[])
{
    std::string macro = argc > 1 ? argv[1] : "TBrun.mac";

    std::unique_ptr<G4RunManager> run_manager{
        G4RunManagerFactory::CreateRunManager(G4RunManagerType::Default)};

    run_manager->SetUserInitialization(new DetectorConstruction{});

    auto& tmi = TMI::Instance();
    auto* physics_list = new FTFP_BERT{/* verbosity = */ 0};
    physics_list->RegisterPhysics(
        new celeritas::TrackingManagerConstructor(&tmi));
    run_manager->SetUserInitialization(physics_list);
    run_manager->SetUserInitialization(new ActionInitialization{});

    tmi.SetOptions(MakeOptions());

    run_manager->Initialize();
    G4UImanager::GetUIpointer()->ApplyCommand("/control/execute " + macro);

    return 0;
}
```

Create a matching `CMakeLists.txt` that follows the current upstream example style.

```cmake
# %%writefile CMakeLists.txt
cmake_minimum_required(VERSION 3.18...4.1)
project(tile_gpu LANGUAGES CXX)

find_package(Celeritas 0.6 REQUIRED)
find_package(Geant4 REQUIRED)

if(NOT CELERITAS_USE_Geant4)
  message(FATAL_ERROR "This Celeritas installation was not built with Geant4 support")
endif()

add_executable(tile_gpu tile_gpu.cc)
target_compile_features(tile_gpu PRIVATE cxx_std_17)

celeritas_target_link_libraries(tile_gpu
  Celeritas::G4
  ${Geant4_LIBRARIES}
)
```

Using `0.6` here keeps the example compatible with current stable Celeritas installs while still matching the tracking-manager integration API used in this notebook.

---

## 4 Configure and build

```bash
# %%bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh
set -e

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The build uses CMake plus `find_package(Celeritas ...)` / `find_package(Geant4 ...)`, both resolved from the Key4hep `CMAKE_PREFIX_PATH`.

If your Celeritas build includes MPI support and you only want a single-process notebook run, set `CELER_DISABLE_PARALLEL=1` before executing the benchmark cells.

---

## 5 Benchmark CPU-only versus GPU-enabled runs

```python
import os
import pathlib
import subprocess
import time


KEY4HEP_SETUP = "/cvmfs/sw.hsf.org/key4hep/setup.sh"


def run_tile(macro: str, use_gpu: bool):
    env = os.environ.copy()
    env.setdefault("CELER_DISABLE_PARALLEL", "1")
    if not use_gpu:
        env["CELER_DISABLE_DEVICE"] = "1"
    else:
        env.pop("CELER_DISABLE_DEVICE", None)

    # Run inside a shell that sources Key4hep so the Geant4/Celeritas runtime
    # libraries (from CVMFS) are on the library path, regardless of how Jupyter
    # was launched. CELER_DISABLE_DEVICE is honoured at runtime.
    cmd = f'source "{KEY4HEP_SETUP}" >/dev/null 2>&1; exec ./build/tile_gpu "{macro}"'
    t0 = time.perf_counter()
    result = subprocess.run(
        ["bash", "-c", cmd],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    dt = time.perf_counter() - t0
    return dt, result.stdout, result.stderr


cpu_t, cpu_out, cpu_err = run_tile("TBrun.mac", use_gpu=False)
gpu_t, gpu_out, gpu_err = run_tile("TBrun.mac", use_gpu=True)

print(f"CPU wall time: {cpu_t:.2f} s")
print(f"GPU wall time: {gpu_t:.2f} s")
if gpu_t > 0:
    print(f"Speed-up: {cpu_t / gpu_t:.2f}x")
```

Notes:

- `TBrun.mac` currently runs a single 18 GeV `pi+` configuration upstream.
- `CELER_DISABLE_DEVICE=1` is the supported way to force CPU mode for a comparison run.
- The total speed-up depends strongly on how much of the workload is electromagnetic versus hadronic.

---

## 6 Inspect the Celeritas diagnostics output

The example above writes `tile_gpu.out.json`. Its exact contents depend on the Celeritas version and enabled diagnostics, so the safest first step is to inspect the top-level structure rather than hard-code field names.

```python
import json
import pathlib
from pprint import pprint

path = pathlib.Path("tile_gpu.out.json")
if not path.exists():
    raise FileNotFoundError("Run the executable first so tile_gpu.out.json exists")

with path.open() as handle:
    diagnostics = json.load(handle)

print("Top-level keys:")
for key in sorted(diagnostics):
    print(" -", key)

print("\nSelected preview:")
preview = {key: diagnostics[key] for key in list(diagnostics)[:3]}
pprint(preview)
```

If you want a deeper analysis, use this JSON as the starting point for extracting transport counts, kernel timings, or build/runtime configuration details.

---

## 7 Exercises

### Exercise 1: Mixed-particle macro

Run `TBrun_all.mac`, which contains electron, pion, kaon, and proton runs from the upstream TileCal repository.

Questions:

1. Does the overall GPU speed-up decrease compared with `TBrun.mac`?
2. Why is that expected for a workload with a larger hadronic component?

```python
cpu_t, _, _ = run_tile("TBrun_all.mac", use_gpu=False)
gpu_t, _, _ = run_tile("TBrun_all.mac", use_gpu=True)
print(f"TBrun_all.mac speed-up: {cpu_t / gpu_t:.2f}x")
```

### Exercise 2: Event-count scaling

Make a copy of `TBrun.mac`, increase the event count, and see whether the GPU run benefits more from the larger workload.

```python
from pathlib import Path

source = Path("TBrun.mac").read_text()
Path("TBrun_100k.mac").write_text(source.replace("/run/beamOn 10000", "/run/beamOn 100000"))

cpu_t, _, _ = run_tile("TBrun_100k.mac", use_gpu=False)
gpu_t, _, _ = run_tile("TBrun_100k.mac", use_gpu=True)
print(f"TBrun_100k.mac speed-up: {cpu_t / gpu_t:.2f}x")
```

Expected discussion point: GPUs usually amortize setup costs better as the problem size grows, but hadronic-heavy transport can still cap the gain.

---

## 8 AdePT — a second GPU EM-transport engine

Celeritas above offloaded EM transport by *replacing* Geant4's transport on the GPU. **[AdePT](https://github.com/apt-sim/AdePT)** takes a complementary route: it is a **Geant4 plugin** that keeps Geant4 in charge and offloads the e⁻/e⁺/γ shower to the GPU via **G4HepEm** physics on **VecGeom** geometry.

AdePT ships a standalone Geant4 application, `example1`, run straight from CVMFS. Source the LCG `devAdePT` view (separate from Key4hep):

```bash
source /cvmfs/sft.cern.ch/lcg/views/devAdePT/latest/x86_64-el9-gcc13-opt/setup.sh
```

The factored helper `06/adept-demo/adept_demo.py` sources that view and provides `verify_adept()` and `run_example1(run=...)`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("adept-demo").resolve()))
import adept_demo as ad

ad.verify_adept()
# On a GPU node, launch the GPU EM-transport example and time it:
#     ad.run_example1(macro="example1.mac", run=True)
ad.run_example1(run=False)
```

`example1` runs the EM shower on the GPU; a standard-Geant4 build without the AdePT physics constructor is the CPU baseline — the same CPU-vs-GPU contrast the Celeritas cells measured with `CELER_DISABLE_DEVICE`. If `example1` is not prebuilt in the view, the helper prints the one-time `git clone` + `cmake` build commands. Requirements (all in the `devAdePT` view): Geant4 > 11, VecGeom ≥ 2.0.0-rc.4, G4HepEm, CUDA > 12, C++20.

---

## 9 Where you can contribute

GPU-accelerated detector simulation is an active, open R&D area and welcomes newcomers. All four projects are open source and take external pull requests.

| Project | Role in the GPU stack | Good first contributions | Links |
| --- | --- | --- | --- |
| **Celeritas** | GPU transport library + Geant4 offload | reproduce/report benchmarks, validation plots, docs, physics tests | [repo](https://github.com/celeritas-project/celeritas) · [issues](https://github.com/celeritas-project/celeritas/issues) |
| **AdePT** | Geant4 plugin offloading EM transport | run `example1` on new geometries, profile kernels, add examples, docs | [repo](https://github.com/apt-sim/AdePT) · [issues](https://github.com/apt-sim/AdePT/issues) |
| **G4HepEm** | Compact EM physics used by AdePT | add/verify physics processes, unit tests, table validation | [repo](https://github.com/mnovak42/g4hepem) |
| **VecGeom** | Vectorized/GPU geometry used by both | geometry unit tests, shape implementations, benchmarks | [CERN GitLab](https://gitlab.cern.ch/VecGeom/VecGeom) |

Ways to start: reproduce a CPU-vs-GPU speed-up for your detector/particle mix, pick a "good first issue" on Celeritas or AdePT, try your own GDML geometry and report where GPU transport helps most (EM showers) and least (hadronic-heavy events), or improve docs and examples.

---

## Further reading

- Celeritas user documentation: https://celeritas-project.github.io/celeritas/user/index.html
- Celeritas Geant4 examples: https://github.com/celeritas-project/celeritas/tree/main/example/geant4
- AdePT project and examples: https://github.com/apt-sim/AdePT
- DD4hep project documentation: https://dd4hep.web.cern.ch/dd4hep/
- Geant4 documentation portal: https://geant4.web.cern.ch/documentation
- TileCal geometry repository used for this lesson: https://github.com/celeritas-project/atlas-tilecal-integration

---

## 10 Clean-up

```bash
# %%bash --no-raise-error
rm -rf build
rm -f tile_gpu.cc CMakeLists.txt tile_gpu.out.json TBrun.mac TBrun_all.mac TBrun_100k.mac
```
