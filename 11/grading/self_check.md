# Self-Check — run this on yourself before submitting

A fast checklist for self-paced learners. If you can't tick a box honestly, go
back to that phase. This is the same standard the grading guide applies, phrased
as questions you ask yourself.

## Before you asked the AI for anything
- [ ] I wrote the kernel contract (inputs, outputs, launch config, memory model)
      **before** generating code.
- [ ] I named the single metric I'd use to judge success.

## Before you ran the generated code
- [ ] I wrote down a numeric prediction for occupancy and speedup, with reasoning.
- [ ] I decided whether I expected it to be memory-bound or compute-bound — and why.
- [ ] I read the generated kernel and listed its correctness risks against the
      failure-mode list (bounds check, sync, races, coalescing, changed math).

## Correctness — the non-negotiable part
- [ ] I have a CPU reference (a plain CPU version or a known-good version) to compare against.
- [ ] I wrote the test harness myself; I did not trust the AI's own check.
- [ ] I tested an event count that is **not** a multiple of the block size.
- [ ] I checked output range and max absolute error, not just "looks fine."
- [ ] The correctness gate prints `PASS`.

## Diagnosis — the part that proves you learned something
- [ ] I have real profiler numbers, not guesses.
- [ ] I can explain the gap between my prediction and the measurement.
- [ ] I can name the current bottleneck in one sentence.
- [ ] My next optimization is tied to a specific concept from Modules 01–03.

## Honesty check
- [ ] My workbook contains at least one "I expected X but got Y" moment.
- [ ] I can explain every optimization to someone else without the AI present.
- [ ] If you deleted the AI transcript, my workbook would still show *I*
      understood what happened and why.

> If the last box isn't true, you used the AI as a crutch, not a tool. The whole
> point of this module is that you can still stand up when it's taken away.
