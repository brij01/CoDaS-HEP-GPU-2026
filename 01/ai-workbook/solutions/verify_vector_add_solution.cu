/******************************************************************************
 * SOLUTION to the Phase-4 verification harness. This is the completed version
 * of verify_vector_add.cu with the launch, error check, and synchronization
 * filled in. It compares the GPU result against a CPU reference and prints
 * PASS/FAIL. N is deliberately NOT a multiple of the block size.
 *
 *   nvcc -arch=native verify_vector_add_solution.cu -o verify && ./verify
 *****************************************************************************/

#include <stdio.h>
#include <math.h>

// ---- Kernel under test: grid-stride vector add with a bounds check. ----------
__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride)
    result[i] = a[i] + b[i];
}

// ---- CPU reference: the source of truth. -------------------------------------
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
  const int N = 1'000'003;                 // NOT a multiple of the block size
  size_t size = N * sizeof(float);

  float *a, *b, *c, *ref;
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);
  ref = (float *)malloc(size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  addVectorsInto_cpu(ref, a, b, N);

  // ---- Launch the kernel under test. -----------------------------------------
  size_t threads = 256;
  size_t blocks = (N + threads - 1) / threads;
  addVectorsInto<<<blocks, threads>>>(c, a, b, N);

  cudaError_t err = cudaGetLastError();      // catch launch errors
  if (err != cudaSuccess) { printf("launch error: %s\n", cudaGetErrorString(err)); return 1; }
  cudaDeviceSynchronize();                    // the sync the AI may have dropped

  // ---- Correctness gate. -----------------------------------------------------
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
