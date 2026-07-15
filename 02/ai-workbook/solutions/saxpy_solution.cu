/******************************************************************************
 * SOLUTION — corrected SAXPY on unified memory, with prefetching and a
 * full-array correctness gate against a CPU reference.
 *
 * The bug in adversarial_saxpy_buggy.cu was the thread index:
 *     int tid = blockIdx.x * blockDim.x * threadIdx.x;   // '*' is wrong
 * It must be:
 *     int tid = blockIdx.x * blockDim.x + threadIdx.x;   // '+'
 *
 * This version also uses a grid-stride loop and cudaMemPrefetchAsync so the
 * pages are resident on the device before the kernel runs (fewer page faults).
 *
 *   nvcc -arch=native saxpy_solution.cu -o saxpy && ./saxpy
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

__global__ void saxpy(int *a, int *b, int *c, int n)
{
  int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    c[i] = 2 * a[i] + b[i];
}

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
  saxpy_cpu(a, b, ref, n);

  int device = -1;
  cudaGetDevice(&device);
  // Prefetch inputs to the device and the output buffer too: this migrates the
  // pages up front instead of faulting them in one-by-one during the kernel.
  cudaMemPrefetchAsync(a, size, device);
  cudaMemPrefetchAsync(b, size, device);
  cudaMemPrefetchAsync(c, size, device);

  int threads_per_block = 256;
  int number_of_blocks = (n + threads_per_block - 1) / threads_per_block;

  saxpy<<<number_of_blocks, threads_per_block>>>(a, b, c, n);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { printf("launch error: %s\n", cudaGetErrorString(err)); return 1; }
  cudaDeviceSynchronize();

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
