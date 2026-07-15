/******************************************************************************
 * VERIFICATION HARNESS STUB — Phase 4 for Module 03 streamed processing.
 *
 * This is YOUR test harness. It ships with a correct CPU reference and a
 * full-array correctness gate. You complete the streamed launches and the
 * per-stream ordering, so the verification is something you authored.
 *
 *   nvcc -arch=native verify_streams.cu -o verify && ./verify
 *
 * N is deliberately NOT a multiple of (NSTREAMS * block size).
 *****************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define NSTREAMS 4

__global__ void square_plus_one(const float *in, float *out, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * in[i] + 1.0f;
}

// ---- CPU reference: the source of truth. -------------------------------------
void square_plus_one_cpu(const float *in, float *out, int n)
{
  for (int i = 0; i < n; ++i) out[i] = in[i] * in[i] + 1.0f;
}

int main()
{
  const int N = 4'000'037;                 // NOT a multiple of NSTREAMS*block
  size_t bytes = (size_t)N * sizeof(float);

  float *h_in, *h_out;
  cudaMallocHost(&h_in, bytes);
  cudaMallocHost(&h_out, bytes);
  float *ref = (float *)malloc(bytes);
  for (int i = 0; i < N; ++i) { h_in[i] = (float)(i % 100); h_out[i] = -1.0f; }

  square_plus_one_cpu(h_in, ref, N);       // ground truth

  float *d_in, *d_out;
  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);

  cudaStream_t stream[NSTREAMS];
  for (int s = 0; s < NSTREAMS; ++s) cudaStreamCreate(&stream[s]);

  // ---- TODO (student): build the streamed pipeline. --------------------------
  // For each chunk s:
  //   1. cudaMemcpyAsync H2D on stream[s].
  //   2. launch square_plus_one on stream[s].
  //   3. cudaMemcpyAsync D2H on stream[s]   <-- SAME stream, this is the point.
  // Then synchronize (cudaDeviceSynchronize or each stream) before the check.

  cudaDeviceSynchronize();

  // ---- Correctness gate: full array vs the CPU reference. --------------------
  int first_bad = -1;
  for (int i = 0; i < N; ++i)
    if (fabsf(h_out[i] - ref[i]) > 1e-3f) { first_bad = i; break; }

  if (first_bad < 0)
    printf("PASS: all %d elements match the CPU reference\n", N);
  else
    printf("FAIL: first mismatch at index %d (got %.3f, expected %.3f)\n",
           first_bad, h_out[first_bad], ref[first_bad]);

  for (int s = 0; s < NSTREAMS; ++s) cudaStreamDestroy(stream[s]);
  cudaFreeHost(h_in); cudaFreeHost(h_out); free(ref);
  cudaFree(d_in); cudaFree(d_out);
}
