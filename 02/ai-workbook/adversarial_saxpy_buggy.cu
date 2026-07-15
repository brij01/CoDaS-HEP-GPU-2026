/******************************************************************************
 * ADVERSARIAL EXAMPLE — this SAXPY kernel "an AI wrote for you" is
 * INTENTIONALLY BUGGY. It compiles and its first printed values can look
 * correct, but most of the output array is never written.
 *
 * Build and run:
 *   nvcc -arch=native adversarial_saxpy_buggy.cu -o buggy && ./buggy
 *
 * The Module 02 trap: a subtly wrong thread index. Printing "the first and last
 * few values" is NOT a correctness test — you need a full-array check against a
 * CPU reference.
 *****************************************************************************/

#include <stdio.h>

#define N (2048 * 2048)

__global__ void saxpy(int *a, int *b, int *c)
{
  // BUG: '*' should be '+'. This makes tid mostly 0 (whenever threadIdx.x is 0)
  // or a wildly out-of-stride value otherwise, so the array is not covered.
  int tid = blockIdx.x * blockDim.x * threadIdx.x;

  if (tid < N)
    c[tid] = 2 * a[tid] + b[tid];
}

int main()
{
  int *a, *b, *c;
  int size = N * sizeof(int);

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  for (int i = 0; i < N; ++i) { a[i] = 2; b[i] = 1; c[i] = 0; }

  int threads_per_block = 128;
  int number_of_blocks = (N / threads_per_block) + 1;

  saxpy<<<number_of_blocks, threads_per_block>>>(a, b, c);
  cudaDeviceSynchronize();

  // This "quality check" only inspects 10 of 4 million elements — it lies.
  for (int i = 0; i < 5; ++i) printf("c[%d] = %d, ", i, c[i]);
  printf("\n");
  for (int i = N - 5; i < N; ++i) printf("c[%d] = %d, ", i, c[i]);
  printf("\n");

  cudaFree(a); cudaFree(b); cudaFree(c);
}
