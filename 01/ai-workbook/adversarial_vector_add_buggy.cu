/******************************************************************************
 * ADVERSARIAL EXAMPLE — this kernel "an AI wrote for you" is INTENTIONALLY
 * BUGGY. Your job is to find the bug, prove it, and fix it. Do not just read the
 * fix below and move on.
 *
 * Build and run:
 *   nvcc -arch=native adversarial_vector_add_buggy.cu -o buggy && ./buggy
 * Run it several times — the failure may be intermittent. That intermittency is
 * the lesson.
 *
 * The Module 01 failure mode: the host thread reads results the GPU may not have
 * finished writing. There is NO compiler error, and you may even get the correct
 * answer sometimes. A test that runs once can pass by luck.
 *****************************************************************************/

#include <stdio.h>

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
  {
    result[i] = a[i] + b[i];
  }
}

void initWith(float num, float *a, int N)
{
  for (int i = 0; i < N; ++i) a[i] = num;
}

void checkElementsAre(float target, float *array, int N)
{
  for (int i = 0; i < N; i++)
  {
    if (array[i] != target)
    {
      printf("FAIL: array[%d] = %0.4f does not equal %0.4f\n", i, array[i], target);
      return;
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2 << 20;
  size_t size = N * sizeof(float);

  float *a, *b, *c;
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  size_t threads = 256;
  size_t blocks = (N + threads - 1) / threads;

  addVectorsInto<<<blocks, threads>>>(c, a, b, N);

  /*
   * BUG LIVES HERE. The kernel launch above is asynchronous: it returns to the
   * CPU immediately, before the GPU has finished (or even started) computing.
   * The line below reads `c` from the host right away.
   *
   * TODO (student): identify the single missing call, add it, and explain in
   *   your workbook why the check can still pass "by luck" without it.
   */

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
