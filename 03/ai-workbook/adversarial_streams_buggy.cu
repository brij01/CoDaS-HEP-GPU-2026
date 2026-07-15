/******************************************************************************
 * ADVERSARIAL EXAMPLE — this streamed pipeline "an AI wrote for you" is
 * INTENTIONALLY BUGGY. It compiles and often prints the right answer, but it
 * contains a CROSS-STREAM RACE: each chunk's kernel runs on stream[i], but each
 * chunk's device->host copy is issued on stream[0]. Operations on different
 * streams are not ordered, so the copy for chunk i>0 can run before chunk i's
 * kernel has finished.
 *
 * Build and run (run several times — the failure may be intermittent):
 *   nvcc -arch=native adversarial_streams_buggy.cu -o buggy && ./buggy
 *
 * The Module 03 trap: within one stream, operations are ordered; ACROSS streams
 * they are not. Putting the copy on the wrong stream breaks that ordering.
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#define N (1 << 22)
#define NSTREAMS 4

__global__ void square_plus_one(const float *in, float *out, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * in[i] + 1.0f;
}

int main()
{
  size_t bytes = (size_t)N * sizeof(float);

  float *h_in, *h_out;
  cudaMallocHost(&h_in, bytes);      // pinned, so async copies are truly async
  cudaMallocHost(&h_out, bytes);
  for (int i = 0; i < N; ++i) { h_in[i] = (float)(i % 100); h_out[i] = -1.0f; }

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

    cudaMemcpyAsync(d_in + offset, h_in + offset, cbytes, cudaMemcpyHostToDevice, stream[s]);
    square_plus_one<<<blocks, threads, 0, stream[s]>>>(d_in + offset, d_out + offset, count);

    // BUG: this copy is issued on stream[0], NOT stream[s]. For s>0 it is not
    // ordered after the kernel above, so it can copy stale d_out.
    cudaMemcpyAsync(h_out + offset, d_out + offset, cbytes, cudaMemcpyDeviceToHost, stream[0]);
  }

  cudaDeviceSynchronize();

  // Only spot-checks a few values — misses the racing chunks most of the time.
  printf("h_out[0]=%.1f  h_out[N-1]=%.1f\n", h_out[0], h_out[N - 1]);

  for (int s = 0; s < NSTREAMS; ++s) cudaStreamDestroy(stream[s]);
  cudaFreeHost(h_in); cudaFreeHost(h_out);
  cudaFree(d_in); cudaFree(d_out);
}
