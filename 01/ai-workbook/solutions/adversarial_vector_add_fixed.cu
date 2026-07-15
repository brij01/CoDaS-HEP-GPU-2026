/******************************************************************************
 * SOLUTION to the adversarial exercise. The single missing call was
 * cudaDeviceSynchronize() before the host reads the results.
 *
 * This version also adds an explicit CPU reference so the correctness check is
 * self-contained and prints a clear PASS/FAIL.
 *
 * Build and run (run several times — it should now pass every time):
 *   nvcc -arch=native adversarial_vector_add_fixed.cu -o fixed && ./fixed
 *****************************************************************************/

#include <stdio.h>
#include <math.h>

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)                       // bounds check: N need not be a multiple of block size
  {
    result[i] = a[i] + b[i];
  }
}

void initWith(float num, float *a, int N)
{
  for (int i = 0; i < N; ++i) a[i] = num;
}

// CPU reference: the obvious, trusted version we compare the GPU against.
void addVectorsInto_cpu(float *result, float *a, float *b, int N)
{
  for (int i = 0; i < N; ++i) result[i] = a[i] + b[i];
}

int main()
{
  const int N = 2 << 20;
  size_t size = N * sizeof(float);

  float *a, *b, *c;
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);
  float *ref = (float *)malloc(size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  addVectorsInto_cpu(ref, a, b, N);          // ground truth on the CPU

  size_t threads = 256;
  size_t blocks = (N + threads - 1) / threads;

  addVectorsInto<<<blocks, threads>>>(c, a, b, N);

  // THE FIX: the kernel launch is asynchronous. Wait for the GPU to finish
  // before the host reads `c`. Without this, the check below races the kernel.
  cudaError_t err = cudaGetLastError();      // catch launch errors too
  if (err != cudaSuccess) { printf("launch error: %s\n", cudaGetErrorString(err)); return 1; }
  cudaDeviceSynchronize();

  // Correctness gate against the CPU reference.
  double max_abs_err = 0.0;
  int first_bad = -1;
  for (int i = 0; i < N; ++i)
  {
    double e = fabs((double)c[i] - (double)ref[i]);
    if (e > max_abs_err) max_abs_err = e;
    if (e > 1e-5 && first_bad < 0) first_bad = i;
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
