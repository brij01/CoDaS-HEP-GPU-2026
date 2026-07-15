# Final-project solution — worked example and explanation

> **Read this only after you have tried the loop yourself.** The skill this
> module trains is the *reasoning*, not the final kernel. If you read the answer
> before predicting and measuring, you skip the exact thing being graded.

This folder contains **one** worked answer to the final project:
[mva_infer_optimized.cu](mva_infer_optimized.cu). It is not the only correct
answer — it is an example of what a solid pass through the 5-phase loop produces,
written up the way your own workbook should be.

## The problem, restated

The baseline ([../mva_infer_baseline.cu](../mva_infer_baseline.cu)) runs a tiny
neural network (4 inputs → 8 ReLU nodes → 1 sigmoid) on millions of events, one
thread per event. It is correct but leaves performance on the table. The metric
is **throughput (M events/sec)**; the hard gate is **RESULT: PASS/FAIL** against
the CPU reference.

## The 5-phase loop applied

### Phase 1 — SPECIFY (what I decided before generating anything)

- **Inputs:** `num_events × 4` floats. The baseline stores them **event-major**
  (array-of-structs): event `e`'s four features are contiguous. With one thread
  per event, thread `e` and thread `e+1` then read addresses 4 floats apart —
  **uncoalesced**.
- **Output:** `num_events` floats in `[0, 1]` (sigmoid range).
- **Launch config:** grid-stride loop, 256 threads/block, so a single
  configuration covers any event count and any GPU.
- **Memory model:** the model (mean, istd, weights, biases) is small,
  read-only, and read by *every* thread → put it in `__constant__` memory. The
  input goes to global memory in a **struct-of-arrays** layout so reads coalesce.
- **Success metric:** M events/sec for the full H2D + kernel + D2H round trip,
  same timing window as the baseline.

### Phase 2 — GENERATE (the AI's job)

Ask the assistant for: a constant-memory model, an SoA input layout, and a
grid-stride kernel, *keeping the same math and the same PASS/FAIL gate*. The
result is [mva_infer_optimized.cu](mva_infer_optimized.cu).

### Phase 3 — PREDICT (before running)

- **Bound:** this is heavily **memory-bound**. Each event does only ~40
  floating-point operations but must load 4 input floats. Coalescing the loads
  is therefore the change most likely to matter; the constant-memory model helps
  because it removes repeated global loads of the weights.
- **Expected speedup:** most of the win comes from coalescing and the constant
  cache. On a typical data-center GPU expect a few× on the kernel itself; the
  end-to-end number is capped by the PCIe transfer of the input array, which
  both versions pay.
- **Correctness risks to watch in the generated code** (from the README list):
  - the SoA index `input[i * num_events + idx]` must match how the host filled
    the array — an easy off-by-layout bug;
  - the grid-stride loop must still cover every event when `grid` is clamped;
  - fast-math intrinsics were **deliberately not** used here, because they can
    push the sigmoid past the `1e-5` tolerance — if you add them, you must
    re-check the gate.

### Phase 4 — VERIFY (my test, not the AI's)

The correctness gate is identical to the baseline and lives in the same file:
compare a sample of GPU outputs against `evaluate_cpu` (the CPU reference), and
check the sigmoid range. Crucially, **run an event count that is not a multiple
of the block size** — that is where a dropped bounds check or a grid-stride
off-by-one hides:

```bash
nvcc -O3 -arch=native mva_infer_optimized.cu -o mva_infer_optimized
./mva_infer_optimized 20000000     # round number
./mva_infer_optimized 19999999     # NOT a multiple of the block size
./mva_infer_optimized 1000         # tiny workload
```

All three must print `RESULT: PASS`. A version that only passes on round numbers
is broken.

### Phase 5 — DIAGNOSE (what to measure and conclude)

Profile with Nsight Systems / Compute and compare against your Phase-3
prediction:

- **Achieved occupancy** should be high (256-thread blocks, small register use).
- **Global memory throughput** should be much closer to the hardware peak than
  the baseline, because the loads are now coalesced. That is the evidence the
  SoA change did what you predicted.
- **The new bottleneck** is almost certainly the **host↔device transfer** of the
  input array, not the kernel. That points to the *next* optimization: unified
  memory + prefetch (Module 02) and overlapping transfer with compute using
  streams (Module 03). Those are left as further iterations for your workbook.

## What the AI could get wrong here (and the process caught)

- Silently keeping the AoS layout while *claiming* it coalesced — caught by
  reading the index expression, not the prose.
- Swapping in `__expf`/`__fdividef` "for speed" and breaking the `1e-5`
  tolerance — caught by the gate, which is why the gate is non-negotiable.
- A grid-stride loop that drops the last partial tile when `grid` is clamped —
  caught by the `19999999` run.

## The honest takeaway

The fast kernel is a few lines different from the slow one. The *value* was in
the four decisions in Phase 1 and the three checks in Phase 4 — none of which the
AI can be trusted to make for you. That is the whole point of the module.
