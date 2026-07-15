/******************************************************************************
 * ADVERSARIAL EXAMPLE — this N-body kernel "an AI wrote for you" is
 * INTENTIONALLY BUGGY. It FUSES force computation and position integration into
 * a single kernel. Because every thread reads all bodies' positions to compute
 * its force, but also writes its own position in the same kernel, threads race:
 * some read positions that have already been advanced by other threads.
 *
 * Build and run:
 *   nvcc -arch=native adversarial_nbody_buggy.cu -o buggy && ./buggy
 *
 * The Module 08 trap: the two phases (compute ALL forces, THEN integrate ALL
 * positions) must be separated by a global synchronization — i.e. two kernel
 * launches. Fusing them into one kernel is a whole-grid race.
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

// BUG: computes force AND integrates position in the same kernel. p[j] read by
// thread i may already have been advanced by thread j -> race, wrong physics.
__global__ void force_and_integrate(Body *p, float dt, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float Fx = 0, Fy = 0, Fz = 0;
  for (int j = 0; j < n; ++j) {
    float dx = p[j].x - p[i].x;
    float dy = p[j].y - p[i].y;
    float dz = p[j].z - p[i].z;
    float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
    float invDist = rsqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;
    Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
  }

  p[i].vx += dt * Fx; p[i].vy += dt * Fy; p[i].vz += dt * Fz;

  // Writing position here, while other threads are still reading p[i] above.
  p[i].x += p[i].vx * dt;
  p[i].y += p[i].vy * dt;
  p[i].z += p[i].vz * dt;
}

int main()
{
  const int n = 4096;
  const float dt = 0.01f;
  const int nIters = 10;

  Body *p;
  cudaMallocManaged(&p, n * sizeof(Body));
  srand(1234);
  for (int i = 0; i < n; ++i) {
    p[i].x = 2.0f*rand()/RAND_MAX - 1.0f; p[i].y = 2.0f*rand()/RAND_MAX - 1.0f;
    p[i].z = 2.0f*rand()/RAND_MAX - 1.0f; p[i].vx = p[i].vy = p[i].vz = 0.0f;
  }

  int threads = 256, blocks = (n + threads - 1) / threads;
  for (int iter = 0; iter < nIters; ++iter) {
    force_and_integrate<<<blocks, threads>>>(p, dt, n);
    cudaDeviceSynchronize();
  }

  printf("body[0] = (%.5f, %.5f, %.5f)\n", p[0].x, p[0].y, p[0].z);
  cudaFree(p);
}
