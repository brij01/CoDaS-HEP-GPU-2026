# Module 11 — Learning GPU Programming with AI

> You may use AI and agentic coding assistants as much as you want in this
> course. This module exists to make sure you still **learn everything the
> course teaches** while doing so.

This module is **standalone**. You can work through it on its own, even if you
have not finished Modules 01–10. It teaches you how to learn — and how to prove
you have learned — GPU programming in a world where an AI can write the code for
you. Everything you need is in this folder.

---

## 1. Why this module exists

If an AI can write the CUDA kernel for you, then *typing the kernel* is no longer
the skill worth grading. The skill is **supervising** the machine:

- deciding **what** to build,
- predicting **how** it will behave on real hardware,
- proving it **correct** when thousands of threads run at once, and
- diagnosing **why** it is (or isn't) fast.

Those four abilities are exactly what Modules 01–10 have always taught. This
module just makes them the explicit, graded work, and gives you a repeatable
routine for doing them with an AI at your side.

## 2. The three things an AI cannot do for you

An AI can generate code that *looks* right. It cannot hold these for you:

1. **A hardware mental model.** Threads, blocks, warps, occupancy, and the memory
   hierarchy. Without this you cannot tell whether generated code will be fast,
   and you cannot read a profiler.
2. **Reasoning about performance.** Memory-bound vs compute-bound, memory
   coalescing, overlapping data transfer with compute, and choosing a launch
   configuration. This is what turns "it runs" into "it runs 40× faster."
3. **Correctness under parallelism.** Race conditions, missing synchronization,
   out-of-bounds threads, non-deterministic bugs. These usually do **not** show
   up as compiler errors, and AI models routinely get them wrong.

Everything below is built to protect these three abilities.

## 3. The 5-phase loop (the core method)

Every AI-assisted GPU task in this course follows the same loop. **You own
phases 1, 3, 4, and 5. The AI only owns phase 2.**

| Phase | Owner | What you produce | Which ability it protects |
|-------|-------|------------------|---------------------------|
| **1. SPECIFY** | you | A written contract: inputs/outputs, launch configuration, memory model, and the single number you will judge success by. | hardware mental model |
| **2. GENERATE** | AI | The kernel / code. This is the "free" part. | — |
| **3. PREDICT** | you | *Before running:* expected occupancy, whether it is memory- or compute-bound, and the correctness risks you can already see in the code. | mental model + performance reasoning |
| **4. VERIFY** | you | A test harness *you* wrote that proves the result correct against a **CPU reference** (a simple, obviously-correct version you trust). | correctness under parallelism |
| **5. DIAGNOSE** | you | Profiler evidence explaining the gap between your prediction and reality, and the next optimization to try. | performance diagnosis |

**The graded artifact is your workbook** — the spec, the prediction, the test
harness, and the profiler-backed diagnosis — **never the kernel itself.** A
perfect kernel with an empty workbook fails. A modest kernel with a sharp,
honest workbook passes.

### What "CPU reference" means

Throughout this module we compare every GPU result against a **CPU reference**:
a plain, single-threaded C/C++ (or Python) function that computes the same answer
the slow, obvious way. It is easy to read and easy to trust, so it is the *source
of truth* you check the fast GPU code against. (In software testing this trusted
comparison is sometimes called a "test oracle"; we just call it the CPU
reference.)

## 4. Known ways AI gets CUDA wrong

Treat every AI-generated kernel as **guilty until your test harness proves it
innocent.** The recurring failures to look for:

- **Dropped bounds check** — the kernel indexes past the end of the array when
  the element count is not a multiple of the block size.
- **Missing synchronization** — no `cudaDeviceSynchronize()` before the host
  reads results, or no `__syncthreads()` between shared-memory writes and reads.
- **Silent race condition** — several threads write the same location without an
  atomic operation (common in histogram / reduction code).
- **Uncoalesced memory access** — an array-of-structs layout that makes
  neighbouring threads read scattered addresses instead of adjacent ones.
- **Forgotten error checking** — a launch failure is swallowed because
  `cudaGetLastError()` is never called.
- **"Optimizations" that change the math** — e.g. swapping in `__expf` /
  `__fdividef` without checking the output still matches the reference within
  tolerance.

Can an AI *find* these bugs? Sometimes — if you ask it the right question and it
happens to reason correctly. But it cannot be *relied on* to find them, and it
cannot be held responsible when it misses one. A bug that produces no compiler
error and sometimes even returns the right answer can slip past a single test run
and past an AI review. **Catching it is your job, and a test harness you wrote
and understand is how you do it.**

## 5. How to work through this module

1. **Read this README** (you're doing it).
2. **Copy the workbook template.** [TEMPLATE_ai_workbook.md](TEMPLATE_ai_workbook.md)
   is the blank 5-phase workbook. Make one copy per task (e.g.
   `workbook_v1.md`). This copy is what you fill in and submit.
3. **Do the final project** in [final-project/](final-project/): drive an AI
   through the 5-phase loop to speed up a real, correct-but-slow GPU program,
   recording each iteration in your workbook.
4. **Grade yourself** with [grading/self_check.md](grading/self_check.md) before
   you submit. Instructors use [grading/grading_guide.md](grading/grading_guide.md).
5. **Apply the loop everywhere else.** Every hands-on module now has an
   `ai-workbook/` folder that applies this same loop to that module's topic, as a
   pair of notebooks — a `problem.ipynb` and a `solution.ipynb` (see
   [Module 01's](../01/ai-workbook/README.md), for example).

You can do all of this from the guided notebook
[Session_AI_Native_GPU.ipynb](Session_AI_Native_GPU.ipynb) if you prefer Jupyter.

## 6. The final project

[final-project/mva_infer_baseline.cu](final-project/mva_infer_baseline.cu) is a
real High-Energy-Physics kernel: the single-hidden-layer neural network used in
the LHCb **Allen** trigger to classify collision events (ReLU + sigmoid). The
baseline is **correct but deliberately slow**:

- one thread per event, with the model weights re-read from global memory by
  every thread,
- plain `cudaMalloc` / `cudaMemcpy` with no unified memory,
- a single synchronous kernel with no overlap of transfer and compute,
- standard `expf` / division — no fast-math, constant, or shared memory.

It is self-contained (no external data, one `nvcc` command). It prints its own
metric (**events/sec**) and a **PASS/FAIL** correctness gate built from a CPU
reference in the same file.

**Your job:** drive an AI assistant through the 5-phase loop to make it faster,
using what the course teaches (execution configuration, unified memory,
constant/shared memory, streams, profiling). Every optimization must keep the
correctness gate green. Record every iteration in your workbook.

The layered headroom, roughly in the order the course introduces it:

1. Scale the workload and pick a sensible launch configuration (Module 01).
2. Coalesce the input layout (array-of-structs → struct-of-arrays).
3. Put the model weights in `__constant__` or shared memory (every thread reads
   them).
4. Use fast-math intrinsics — **and prove the output still passes the gate.**
5. Switch to unified memory and prefetch (Module 02).
6. Overlap host↔device transfer with compute using streams (Module 03).
7. Tune block size / occupancy using the profiler (Module 03).

A fully worked, explained optimization is in
[final-project/solutions/](final-project/solutions/). **Try the loop yourself
first** — reading the solution before you predict and measure skips the exact
skill the module is training.

## 7. Files in this module

- [README.md](README.md) — this guide.
- [Session_AI_Native_GPU.ipynb](Session_AI_Native_GPU.ipynb) — the guided
  notebook version of the final project.
- [TEMPLATE_ai_workbook.md](TEMPLATE_ai_workbook.md) — the blank 5-phase
  workbook. Copy it per task.
- [final-project/mva_infer_baseline.cu](final-project/mva_infer_baseline.cu) —
  the correct-but-slow baseline with its built-in CPU reference and PASS/FAIL
  gate.
- [final-project/solutions/](final-project/solutions/) — a worked, explained
  optimized version.
- [grading/grading_guide.md](grading/grading_guide.md) — the AI-resilient grading
  guide (for instructors and for self-assessment).
- [grading/self_check.md](grading/self_check.md) — a fast checklist to run on
  yourself before submitting.

## 8. Build and run the baseline

```bash
cd final-project
nvcc -O3 -arch=native mva_infer_baseline.cu -o mva_infer_baseline
./mva_infer_baseline            # 20,000,000 events
./mva_infer_baseline 50000000   # custom event count
```

The notebook runs the same command through a subprocess, so you can work entirely
inside Jupyter if you prefer.
