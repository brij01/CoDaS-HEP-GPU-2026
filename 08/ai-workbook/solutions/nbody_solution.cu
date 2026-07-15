/******************************************************************************
 * SOLUTION — correct GPU N-body. Force computation and position integration are
 * SEPARATE kernels with a device synchronization between them, so every force is
 * computed from the old positions before any position is advanced.
 *
 *   nvcc -O3 -arch=native nbody_solution.cu -o nbody && ./nbody
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

__global__ void bodyForce(Body *p, float dt, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float Fx = 0, Fy = 0, Fz = 0;
  for (int j = 0; j < n; ++j) {
    float dx = p[j].x - p[i].x, dy = p[j].y - p[i].y, dz = p[j].z - p[i].z;
    float d = dx*dx + dy*dy + dz*dz + SOFTENING;
    float inv = rsqrtf(d); float inv3 = inv*inv*inv;
    Fx += dx*inv3; Fy += dy*inv3; Fz += dz*inv3;
  }
  p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;   // only velocity here
}

__global__ void integrate(Body *p, float dt, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  p[i].x += p[i].vx*dt; p[i].y += p[i].vy*dt; p[i].z += p[i].vz*dt;
}

static void step_cpu(Body *p, float dt, int n)
{
  for (int i = 0; i < n; ++i) {
    float Fx = 0, Fy = 0, Fz = 0;
    for (int j = 0; j < n; ++j) {
      float dx = p[j].x - p[i].x, dy = p[j].y - p[i].y, dz = p[j].z - p[i].z;
      float d = dx*dx + dy*dy + dz*dz + SOFTENING;
      float inv = 1.0f / sqrtf(d); float inv3 = inv*inv*inv;
      Fx += dx*inv3; Fy += dy*inv3; Fz += dz*inv3;
    }
    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
  for (int i = 0; i < n; ++i) {
    p[i].x += p[i].vx*dt; p[i].y += p[i].vy*dt; p[i].z += p[i].vz*dt;
  }
}

int main()
{
  const int n = 4096; const float dt = 0.01f; const int nIters = 10;

  Body *p; cudaMallocManaged(&p, n * sizeof(Body));
  Body *ref = (Body *)malloc(n * sizeof(Body));
  srand(1234);
  for (int i = 0; i < n; ++i) {
    float x = 2.0f*rand()/RAND_MAX - 1.0f, y = 2.0f*rand()/RAND_MAX - 1.0f,
          z = 2.0f*rand()/RAND_MAX - 1.0f;
    p[i]   = (Body){x, y, z, 0, 0, 0};
    ref[i] = (Body){x, y, z, 0, 0, 0};
  }
  for (int iter = 0; iter < nIters; ++iter) step_cpu(ref, dt, n);

  int threads = 256, blocks = (n + threads - 1) / threads;
  for (int iter = 0; iter < nIters; ++iter) {
    bodyForce<<<blocks, threads>>>(p, dt, n);
    cudaDeviceSynchronize();              // all forces before any integration
    integrate<<<blocks, threads>>>(p, dt, n);
    cudaDeviceSynchronize();
  }

  double max_err = 0.0;
  for (int i = 0; i < n; ++i) {
    max_err = fmax(max_err, fabs((double)p[i].x - ref[i].x));
    max_err = fmax(max_err, fabs((double)p[i].y - ref[i].y));
    max_err = fmax(max_err, fabs((double)p[i].z - ref[i].z));
  }
  const double tol = 1e-3;
  printf("max abs position error vs CPU: %.3e (tol %.1e)\n", max_err, tol);
  printf("RESULT: %s\n", max_err <= tol ? "PASS" : "FAIL");

  cudaFree(p); free(ref);
}
