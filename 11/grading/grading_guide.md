# Grading Guide — Learning GPU Programming with AI (final project)

This guide is **AI-resilient**: it rewards the reasoning that AI cannot do for
the student, and it is largely immune to a student who simply pasted a fast
kernel from an assistant. The kernel itself is worth almost nothing; the
workbook is worth almost everything.

**Hard gate:** any submission whose final kernel fails the correctness check
(`RESULT: FAIL`) is capped at 40% regardless of speed. Fast-but-wrong is worth
less than slow-but-right.

Total: 100 points.

| Criterion | Points | What earns full marks | What earns zero |
|-----------|-------:|-----------------------|-----------------|
| **1. SPECIFY — kernel contract** | 15 | Complete contract written *before* generation: I/O shapes and layout, launch config with justification, explicit memory model, and a measurable success metric. | Spec is missing, vague, or was clearly written after the fact to match the code. |
| **2. PREDICT — pre-run reasoning** | 20 | Concrete, numeric predictions of occupancy and speedup with reasoning; correctly identifies memory- vs compute-bound; names specific correctness risks in the generated code. | No predictions, or predictions with no reasoning ("it'll be faster"). |
| **3. VERIFY — student-authored test harness** | 25 | A harness the student wrote (not the AI) that checks against a CPU reference, covers edge cases (non-multiple event counts, boundary values, output range), and is actually run. Bugs in AI output are caught here. | Relies solely on the AI's own check, or only eyeballs a few outputs. |
| **4. DIAGNOSE — profiler-backed analysis** | 25 | Real profiler evidence (Nsight numbers), an honest prediction-vs-reality comparison, correct identification of the current bottleneck, and a justified next step tied to a course concept. | No profiler data; "it got faster" with no explanation of *why*. |
| **5. Iteration & reflection** | 15 | ≥2 optimization iterations logged with measured deltas and PASS status; final reflection honestly names what the AI got wrong and what the student learned. | Single version, no iteration log, or reflection is generic filler. |

## Scoring notes for instructors

- **Detecting empty supervision:** if the workbook's PREDICT and DIAGNOSE phases
  could have been written without ever running the code, score them near zero.
  The tell is the absence of *surprise* — good phase-5 entries almost always
  contain "I expected X but measured Y."
- **Correctness over speed:** a student who reaches 3× with a bulletproof harness
  and a sharp diagnosis outscores a student who reaches 30× but can't explain why
  or whose harness misses the boundary bug.
- **AI-caught bugs are a positive signal.** Reward students whose VERIFY phase
  documents a real bug the AI introduced (dropped bounds check, missing sync,
  changed math). Finding those is the skill.
- **Speed bonus (optional, capped at +5):** award up to 5 discretionary points
  for the largest correct speedup in the cohort, but only if phases 3–5 justify
  how it was achieved. Never award speed on its own.
