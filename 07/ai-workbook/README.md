# Module 07 - AI Workbook

This workbook is split into two notebooks so it is easy to follow:

1. **[problem.ipynb](./problem.ipynb)** - the explanation and the problem, with links to the problem program and cells that reproduce the bug and run your verification harness.
2. **[solution.ipynb](./solution.ipynb)** - the worked fix and explanation, with cells that build and run the solution at the end.

Program files:

- Problem program: [adversarial_fcnn_buggy.py](./adversarial_fcnn_buggy.py)
- Verification harness: [verify_fcnn.py](./verify_fcnn.py)
- Solution program(s):
  - [solutions/fcnn_solution.py](./solutions/fcnn_solution.py)

The exercise runs the main notebook's network (the Allen-style `4 -> 8 ReLU -> 1 sigmoid`
classifier) over a large **batch of events** and asks the Module 11 question: is the
*fast* GPU version still *correct*? The intentional bug is a GPU "optimization" that
normalises inputs with **batch statistics** instead of the model's fixed per-feature
mean/std - fast, right shape, values in `[0, 1]`, and silently wrong. The verifier
compares against a CPU reference and reports the GPU-vs-CPU speedup, so you can see
that "fast" and "correct" are separate questions. It runs on CPU (NumPy) everywhere;
with CuPy and a GPU the same code runs on the GPU and the harness prints the speedup.

Start with **problem.ipynb**. See [Module 11](../../11/README.md) for the 5-phase loop that underpins every AI workbook.
