/******************************************************************************
 * VERIFICATION HARNESS STUB — Phase 4 for Module 02 SAXPY on unified memory.
 *
 * This is YOUR test harness. It ships with a correct CPU reference and a
 * full-array correctness gate. What it does NOT do is launch your kernel — you
 * complete that, so the verification is something you authored and understand.
 *
 *   nvcc -arch=native verify_saxpy.cu -o verify && ./verify
 *
 * N is deliberately NOT a multiple of the block size, because that is where
 * dropped bounds checks and off-by-one grid math hide.
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

// ---- Kernel under test. Paste/adapt the AI-generated kernel here. ------------
__global__ void saxpy(int *a, int *b, int *c, int n)
{
  // TODO (student): implement (or paste the AI's) SAXPY: c[i] = 2*a[i] + b[i].
  // Use a correct thread index and a bounds check against n.
}

// ---- CPU reference: the source of truth. Do not change this. -----------------
void saxpy_cpu(int *a, int *b, int *c, int n)
{
  for (int i = 0; i < n; ++i) c[i] = 2 * a[i] + b[i];
}

int main()
{
  const int n = 4'000'037;                 // NOT a multiple of the block size
  int size = n * sizeof(int);

  int *a, *b, *c, *ref;
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);
  ref = (int *)malloc(size);

  for (int i = 0; i < n; ++i) { a[i] = 2; b[i] = 1; c[i] = 0; }

  saxpy_cpu(a, b, ref, n);                 // ground truth

  // ---- TODO (student): launch the kernel under test. -------------------------
  // 1. Choose threads_per_block and number_of_blocks that cover n.
  // 2. (Optional) cudaMemPrefetchAsync a, b, c to the device first.
  // 3. Launch saxpy<<<...>>>(a, b, c, n).
  // 4. cudaGetLastError() to catch launch failures.
  // 5. cudaDeviceSynchronize() so the host waits for the GPU.

  // ---- Correctness gate: full array vs the CPU reference. --------------------
  int first_bad = -1;
  for (int i = 0; i < n; ++i)
    if (c[i] != ref[i]) { first_bad = i; break; }

  if (first_bad < 0)
    printf("PASS: all %d elements match the CPU reference\n", n);
  else
    printf("FAIL: first mismatch at index %d (got %d, expected %d)\n",
           first_bad, c[first_bad], ref[first_bad]);

  cudaFree(a); cudaFree(b); cudaFree(c); free(ref);
}
