# AI Workbook — 5-Phase Loop

> Copy this file for each AI-assisted task (e.g. `workbook_v1.md`). This
> document — not the kernel — is what gets graded. Fill every phase in your own
> words. Pasting AI output into a phase without your own reasoning scores zero
> for that phase.

**Task:** _one line — what are you accelerating?_
**Baseline metric:** _e.g. baseline throughput = ___ M events/sec, from the baseline program_
**Target metric:** _the number you're trying to beat_
**Date / iteration:** _______

---

## Phase 1 — SPECIFY (you)

Write the kernel contract **before** asking the AI for anything.

- **Inputs** (shapes, dtypes, layout — AoS or SoA?):
- **Outputs** (shape, dtype, valid range):
- **Launch configuration** you intend (grid, block, why):
- **Memory model** (global / constant / shared / unified? where does each array live?):
- **Success metric** (the single number you'll judge this by):
- **Correctness definition** (what makes an output correct? tolerance vs the CPU reference?):

## Phase 2 — GENERATE (AI)

- **Prompt you gave the AI** (paste it):
- **What the AI produced** (filename / brief description — do NOT paste the whole kernel here, link to the file):
- **Anything the AI assumed that you did not ask for:**

## Phase 3 — PREDICT (you) — *before running*

Commit to predictions in writing. Being wrong here is fine and expected; not
predicting is not.

- **Occupancy / launch:** how many threads, how many warps per block, expected occupancy?
- **Bound:** do you expect this to be memory-bound or compute-bound? Why?
- **Expected speedup vs baseline** (a number, with reasoning):
- **Correctness risks you can already see in the generated code** (check against the LLM failure-mode list — bounds check? sync? races? coalescing? math changes?):

## Phase 4 — VERIFY (you)

The AI does not get to certify its own work. You do.

- **CPU reference** (what are you comparing against — the plain CPU version? a previous version?):
- **Test harness** (describe the check *you* wrote; link to it. It must cover edge
  cases: event count not a multiple of block size, boundary values, range checks):
- **Result of the correctness gate** (PASS/FAIL, max abs error):
- **Edge cases you deliberately tested:**

## Phase 5 — DIAGNOSE (you)

- **Measured metric** (actual throughput / time):
- **Prediction vs reality** — did it match Phase 3? If not, why were you wrong?
- **Profiler evidence** (Nsight Systems / Compute: paste the key numbers — achieved
  occupancy, memory throughput, stall reasons, transfer vs compute time):
- **The bottleneck now** (name it):
- **Next optimization** you'll try, and which course concept it comes from:

---

### Iteration log

| Version | Change made | Throughput | PASS? | Bottleneck found |
|---------|-------------|-----------|-------|------------------|
| baseline | — | | | |
| v1 | | | | |
| v2 | | | | |
| v3 | | | | |

### Final reflection

- What did the AI get **wrong** that your process caught?
- Which optimization gave the biggest win, and did you predict that in advance?
- If you had no AI, which phase would have taken you longest — and what does that
  tell you about the skill you most need to build?
