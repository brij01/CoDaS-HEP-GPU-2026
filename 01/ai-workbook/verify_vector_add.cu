/******************************************************************************
 * VERIFICATION HARNESS STUB — Phase 4 of the loop for Module 01 vector add.
 *
 * This is YOUR test harness, not the AI's. It ships with a correct CPU reference
 * and a working correctness check. What it does NOT do is run your GPU kernel —
 * that's the part you complete, so that the verification is something you
 * authored and understand.
 *
 * Build and run once you've filled in the TODOs:
 *   nvcc -arch=native verify_vector_add.cu -o verify && ./verify
 *
 * The harness deliberately uses an element count that is NOT a multiple of the
 * block size, because that is where dropped bounds checks and off-by-one grid
 * math hide.
 *****************************************************************************/

#include <stdio.h>
#include <math.h>

// ---- The kernel under test. Paste/adapt the AI-generated kernel here. --------
__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  // TODO (student): implement (or paste the AI's) grid-stride vector add.
  // Make sure it has a bounds check against N.
}

// ---- CPU reference: the source of truth. Do not change this. -----------------
void addVectorsInto_cpu(float *result, float *a, float *b, int N)
{
  for (int i = 0; i < N; ++i) result[i] = a[i] + b[i];
}

void initWith(float num, float *a, int N)
{
  for (int i = 0; i < N; ++i) a[i] = num;
}

int main()
{
  // Deliberately NOT a power of two / multiple of the block size.
  const int N = 1'000'003;
  size_t size = N * sizeof(float);

  float *a, *b, *c, *ref;
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);
  ref = (float *)malloc(size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  // Ground truth from the CPU reference.
  addVectorsInto_cpu(ref, a, b, N);

  // ---- TODO (student): launch the kernel under test. -------------------------
  // 1. Choose a launch configuration (threads per block, blocks).
  // 2. Launch addVectorsInto<<<...>>>(c, a, b, N).
  // 3. Check for launch errors with cudaGetLastError().
  // 4. Add the synchronization the AI may have dropped, so the host waits for
  //    the GPU before the comparison below.

  // ---- Correctness gate: compare against the CPU reference. ------------------
  double max_abs_err = 0.0;
  int first_bad = -1;
  for (int i = 0; i < N; ++i)
  {
    double err = fabs((double)c[i] - (double)ref[i]);
    if (err > max_abs_err) max_abs_err = err;
    if (err > 1e-5 && first_bad < 0) first_bad = i;
  }

  if (first_bad < 0)
    printf("PASS: max abs error %.3e over %d elements\n", max_abs_err, N);
  else
    printf("FAIL: first mismatch at index %d (got %.4f, expected %.4f)\n",
           first_bad, c[first_bad], ref[first_bad]);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  free(ref);
}
