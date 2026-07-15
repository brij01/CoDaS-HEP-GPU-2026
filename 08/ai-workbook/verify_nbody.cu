/******************************************************************************
 * VERIFICATION HARNESS STUB — Phase 4 for Module 08 N-body.
 *
 * This is YOUR test harness. It ships with a correct CPU reference (separate
 * force and integrate phases) and a correctness gate over ALL bodies. You launch
 * the GPU kernels, so the verification is something you authored.
 *
 *   nvcc -arch=native verify_nbody.cu -o verify && ./verify
 *****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

// ---- GPU kernels under test. -------------------------------------------------
__global__ void bodyForce(Body *p, float dt, int n)
{
  // TODO (student): one thread per body; sum forces from all j; update velocity.
}

__global__ void integrate(Body *p, float dt, int n)
{
  // TODO (student): one thread per body; advance position by velocity*dt.
}

// ---- CPU reference: the source of truth. Two separate phases. ----------------
static void step_cpu(Body *p, float dt, int n)
{
  for (int i = 0; i < n; ++i) {                 // phase 1: all forces from OLD pos
    float Fx = 0, Fy = 0, Fz = 0;
    for (int j = 0; j < n; ++j) {
      float dx = p[j].x - p[i].x, dy = p[j].y - p[i].y, dz = p[j].z - p[i].z;
      float d = dx*dx + dy*dy + dz*dz + SOFTENING;
      float inv = 1.0f / sqrtf(d); float inv3 = inv*inv*inv;
      Fx += dx*inv3; Fy += dy*inv3; Fz += dz*inv3;
    }
    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
  for (int i = 0; i < n; ++i) {                 // phase 2: THEN integrate
    p[i].x += p[i].vx*dt; p[i].y += p[i].vy*dt; p[i].z += p[i].vz*dt;
  }
}

int main()
{
  const int n = 4096;
  const float dt = 0.01f;
  const int nIters = 10;

  Body *p; cudaMallocManaged(&p, n * sizeof(Body));
  Body *ref = (Body *)malloc(n * sizeof(Body));
  srand(1234);
  for (int i = 0; i < n; ++i) {
    float x = 2.0f*rand()/RAND_MAX - 1.0f, y = 2.0f*rand()/RAND_MAX - 1.0f,
          z = 2.0f*rand()/RAND_MAX - 1.0f;
    p[i]   = (Body){x, y, z, 0, 0, 0};
    ref[i] = (Body){x, y, z, 0, 0, 0};
  }

  for (int iter = 0; iter < nIters; ++iter) step_cpu(ref, dt, n);  // ground truth

  int threads = 256, blocks = (n + threads - 1) / threads;
  for (int iter = 0; iter < nIters; ++iter) {
    // ---- TODO (student): launch the TWO phases with a sync between them. ------
    // bodyForce<<<blocks, threads>>>(p, dt, n);
    // cudaDeviceSynchronize();               // all forces done before integrate
    // integrate<<<blocks, threads>>>(p, dt, n);
    // cudaDeviceSynchronize();
    (void)blocks; (void)threads;
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
