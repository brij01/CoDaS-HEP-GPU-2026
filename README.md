<!--
README Documentation Comment

This README provides an overview of the CoDaS-HEP 2026, Princeton repository, which contains tutorials, demos, and analysis scripts for GPU programming prepared for the CoDaS-HEP 2026 school at Princeton. It details the repository structure, including modules on CUDA basics, unified memory, streaming, profiling, advanced topics, neural network demos, analysis scripts, and notebook verification tools. The README also lists available presentations, describes the contents and purpose of each module, and provides instructions for getting started and using the verification script. Attribution for external code and licensing information are included. The document is intended to guide users through the repository's resources and facilitate learning and analysis of GPU programming techniques.
-->
# CoDaS-HEP 2026, Princeton

This repository provides tutorials, demos, and analysis scripts for GPU programming, prepared for the CoDaS-HEP 2026 school at Princeton. Materials are organized into modules covering CUDA basics, unified memory, streaming, profiling, HEP physics usecases and neural network demos.


Presentations:
- Session 1 - https://1drv.ms/p/c/5a70ac10b7f66de0/IQBCddvYryMcSp1M0Jffv3c9AeNlQ0UZgff66wyW6UIJGjo?e=ikThrV


- Session 2 - https://1drv.ms/p/c/5a70ac10b7f66de0/IQBH1OhQ9qg5R5sCR0QUaKdVAbZsH2aH8r2MMfdNJ6CyZL0?e=UqZBwF

- Session 3 - https://1drv.ms/p/c/5a70ac10b7f66de0/IQCtJa6uHPUuRqwlhDJHAQitASNoc_xlrQrWeOY9nYD7Gk8?e=cPUO13



## Repository Structure

- **01/** – Introductory CUDA C notebooks and exercises  
  - Presentations (`AC_CUDA_C_*.pptx`)
  - Main tutorial notebook: `Session1.ipynb`
  - Subfolders:  
    - `01-hello/` to `09-heat/`: Step-by-step CUDA examples (hello world, parallelism, indices, loops, memory allocation, error handling, vector addition, matrix multiplication, heat conduction)  
    - `edit/`: Editable files for exercises

- **02/** – Unified Memory tutorials  
  - Main notebook: `Session2_advanced.ipynb`
  - Subfolders: Vector addition, device properties, page faults, prefetching, SAXPY example, and editable files

- **03/** – Streaming and Visual Profiling  
  - Main notebook: `Streaming and Visual Profiling.ipynb`
  - Subfolders: Vector addition, kernel initialization, prefetch checks, stream introduction, manual memory allocation, overlap transfer, n-body simulation, and editable files

- **04/** – Python on GPUs and HEP examples  
  - Main walkthrough notebook: `Session3_Python-GPU_HEP.ipynb`
  - Practice notebooks: `lesson-4-workbook.ipynb`, `lesson-4-project.ipynb`
  - Supporting assets: `data/`, `img/`

- **05/** – Particle physics generators and GPU-friendly analysis  
  - Notebook: `generator_tutorial_gpu_annotated.ipynb`

- **06/** – GPU-accelerated Geant4, TileCal geometry, and Celeritas  
  - Notebook: `gpu_dd4hep_tilecal.ipynb`
  - Documentation: `gpu_geant4_dd4hep_cuda_notebook.md`
  - Geometry input: `TileTB_2B1EB_nobeamline.gdml`

- **07/** – Neural Network demo walkthrough and supporting code  
  - Walkthrough notebook: `cuda_neural_network_demo_complete.ipynb`
  - Script: `verify_notebook.py`
  - Demo source tree: `gpu-demo/`

- **08/** – GPU challenge: N-body optimization exercise  
  - Code: `01-nbody.cu`
  - Notebook: `GPU challenge.ipynb`

- **09/** – Ray tracing, RT hardware, and physics transport tutorial  
  - Notebook: `RT_cores_ray_tracing_tutorial.ipynb`
  - Editable examples in `edit/`: CPU ray-tracing baseline, NVIDIA-style CUDA ray tracer, and a minimal OptiX project skeleton
  - Covers a CPU ray-tracing baseline, RT acceleration concepts, external runnable example files, side-by-side NVIDIA/AMD/Intel ecosystem notes, and a physics-oriented photon propagation example for distant-galaxy light transport

- **10/** – Accelerated Python (NVIDIA Accelerated Computing Hub)  
  - `Accelerated_Python_User_Guide/`: chapter-by-chapter guide to GPU-accelerated Python
  - Supporting material: `tutorials/`, `resources/`, `docs/`, `brev/`, and `events/`

- **11/** – Learning GPU Programming with AI (supervising AI while still learning the material)  
  - Standalone guide and guided notebook: `README.md`, `Session_AI_Native_GPU.ipynb`
  - The five-phase loop (SPECIFY → GENERATE → PREDICT → VERIFY → DIAGNOSE) where the student owns everything except code generation
  - Self-contained HEP final project: `final-project/mva_infer_baseline.cu` (correct-but-slow LHCb Allen MVA trigger kernel to profile and accelerate) with a worked answer in `final-project/solutions/`
  - Reusable workbook `TEMPLATE_ai_workbook.md` and AI-resilient grading in `grading/`
  - Every hands-on module (01–10) has an `ai-workbook/` companion that applies this supervision layer to that module's topic, each with an intentionally buggy AI-written example, a verification harness with a CPU reference, and a worked `solutions/` folder

### AI workbooks (per module)

Each `NN/ai-workbook/` folder is a short companion that reuses the Module 11 loop. It is organised as two student notebooks plus the program files they run:

- **`problem.ipynb`** — the explanation and the problem, with links to the problem program and cells that reproduce the bug and run your verification harness.
- **`solution.ipynb`** — the worked fix and explanation, with cells that build and run the solution at the end.
- the **problem program** (an AI-written example that compiles/runs but is subtly wrong), a **`verify_*` harness** with a built-in CPU reference and PASS/FAIL gate that defaults to FAIL until you complete it, and a **`solutions/`** directory with the corrected program.

Start each with `problem.ipynb`. The intentional bug per module:

| Module | Topic | Intentional bug demonstrated |
|--------|-------|------------------------------|
| 01 | vector add | missing `cudaDeviceSynchronize()` (host reads before GPU finishes) |
| 02 | unified memory / SAXPY | wrong thread index (`*` vs `+`); prefetch lesson |
| 03 | streams & profiling | cross-stream race (copy on the wrong stream) |
| 04 | Python on GPUs / HEP | histogram scatter without accumulation (GPU: no `atomicAdd`) |
| 05 | generators / analysis | `float32` accumulation precision loss |
| 06 | Geant4 / Celeritas | calorimeter energy scatter-add race |
| 07 | neural network demo | GPU inference normalized with batch statistics instead of the model's fixed mean/std (fast but wrong) |
| 08 | N-body | fused force + integrate in one kernel (whole-grid race) |
| 09 | ray tracing | wrong intersection root (far instead of near) |
| 10 | accelerated Python | in-place stencil without double-buffering |



## Notebooks

Each module contains Jupyter notebooks (`.ipynb`) with explanations, code samples, and exercises. Topics include:

- CUDA programming basics
- Memory management (device, unified memory)
- Parallel algorithms (vector/matrix operations, heat conduction)
- Streaming and concurrency
- Profiling and performance analysis
- HEP event generation and analysis
- Ray tracing hardware and photon transport modeling
- Neural network demos


## Scripts

- [`07/verify_notebook.py`](07/verify_notebook.py):  
  Analyze notebook structure, extract code cells to files, and report statistics on content and coverage.

## Credits

- The heat conduction CPU source code in `01/Session1.ipynb` is credited to [An OpenACC Example Code for a C-based heat conduction code](http://docplayer.net/30411068-An-openacc-example-code-for-a-c-based-heat-conduction-code.html) from the University of Houston.

## Getting Started

1. Clone the repository.
2. Verify your GPU toolchain and Python GPU packages:
     ```sh
    python check_gpu.py
     ```
3. Open the notebooks in Jupyter or VS Code.
4. Follow the step-by-step exercises in each module.
5. Use the verification script to analyze or extract code from notebooks:
     ```sh
    python 07/verify_notebook.py <notebook_path>
     ```

## License

The **original course material** in this repository is licensed under the
**Creative Commons Attribution 4.0 International License (CC BY 4.0)** — see
[LICENSE](LICENSE). Copyright © 2025 Brij Kishor Jashal. CC BY 4.0 fits this
repository's mostly-educational content (notebooks, slides, write-ups); reuse is
allowed with attribution.

This repository also bundles or adapts third-party material that keeps its **own**
license — it is *not* relicensed under CC BY 4.0. See [NOTICE](NOTICE) for full
attribution. In short:

- **Modules 01–03** — adapted from **NVIDIA Deep Learning Institute** ("Fundamentals
  of Accelerated Computing with CUDA C/C++" and related). © NVIDIA Corporation;
  NVIDIA's terms apply.
- **Module 10** (`10/`, git submodule of NVIDIA/accelerated-computing-hub) —
  **Apache-2.0** for code ([10/LICENSE-CODE](10/LICENSE-CODE)) and
  **CC BY-NC-SA 4.0** (non-commercial) for content ([10/LICENSE](10/LICENSE)).
- **`07/gpu-demo/`** — derived from **LHCb Allen**, © CERN, Apache-2.0 (header
  retained in the sources).
- **Heat-conduction example** (`01/`) — credited to a University of Houston OpenACC example.

> Because of the NVIDIA DLI material and the non-commercial (CC BY-NC-SA 4.0)
> content in module 10, the repository **as a whole is not uniformly
> permissive/commercial-friendly**. Check the individual licenses before reuse.

---

For more details, explore the subfolders and notebooks in each module.