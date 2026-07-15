/******************************************************************************
 * SOLUTION — correct streamed pipeline. Each chunk's H2D copy, kernel, and D2H
 * copy all run on the SAME stream, so they are ordered. Different chunks use
 * different streams, so they overlap.
 *
 *   nvcc -arch=native streams_solution.cu -o streams && ./streams
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

void square_plus_one_cpu(const float *in, float *out, int n)
{
  for (int i = 0; i < n; ++i) out[i] = in[i] * in[i] + 1.0f;
}

int main()
{
  const int N = 4'000'037;
  size_t bytes = (size_t)N * sizeof(float);

  float *h_in, *h_out;
  cudaMallocHost(&h_in, bytes);
  cudaMallocHost(&h_out, bytes);
  float *ref = (float *)malloc(bytes);
  for (int i = 0; i < N; ++i) { h_in[i] = (float)(i % 100); h_out[i] = -1.0f; }
  square_plus_one_cpu(h_in, ref, N);

  float *d_in, *d_out;
  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);

  cudaStream_t stream[NSTREAMS];
  for (int s = 0; s < NSTREAMS; ++s) cudaStreamCreate(&stream[s]);

  int chunk = (N + NSTREAMS - 1) / NSTREAMS;
  int threads = 256;

  for (int s = 0; s < NSTREAMS; ++s) {
    int offset = s * chunk;
    int count = (offset + chunk <= N) ? chunk : (N - offset);
    if (count <= 0) break;
    size_t cbytes = (size_t)count * sizeof(float);
    int blocks = (count + threads - 1) / threads;

    // ALL THREE on stream[s]: ordered relative to each other, overlapping across s.
    cudaMemcpyAsync(d_in + offset, h_in + offset, cbytes, cudaMemcpyHostToDevice, stream[s]);
    square_plus_one<<<blocks, threads, 0, stream[s]>>>(d_in + offset, d_out + offset, count);
    cudaMemcpyAsync(h_out + offset, d_out + offset, cbytes, cudaMemcpyDeviceToHost, stream[s]);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { printf("launch error: %s\n", cudaGetErrorString(err)); return 1; }
  cudaDeviceSynchronize();

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
