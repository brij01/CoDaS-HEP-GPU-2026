/******************************************************************************
 * Module 11 final-project SOLUTION: an optimized MVA trigger inference kernel.
 *
 * This is ONE worked answer to the final project. Read solutions/README.md for
 * the 5-phase reasoning that produced it. Try the loop yourself before reading
 * this file — the point of the module is the reasoning, not the code.
 *
 * What changed vs final-project/mva_infer_baseline.cu, and why:
 *   1. Struct-of-arrays (SoA) input layout so neighbouring threads read
 *      adjacent addresses -> coalesced global loads.
 *   2. Model parameters in __constant__ memory (every thread reads them; they
 *      are small and read-only -> ideal for the constant cache).
 *   3. Grid-stride loop so one launch config handles any event count and any
 *      GPU, with good occupancy.
 *   4. Same CPU reference and the same PASS/FAIL gate as the baseline, so the
 *      speedup is only "real" if correctness still holds.
 *
 * Self-contained. Compile and run:
 *   nvcc -O3 -arch=native mva_infer_optimized.cu -o mva_infer_optimized
 *   ./mva_infer_optimized             # default 20,000,000 events
 *   ./mva_infer_optimized 19999999    # NOT a multiple of the block size
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>

#define NUM_INPUT 4
#define NUM_NODE  8

struct Model {
    float mean[NUM_INPUT];
    float istd[NUM_INPUT];
    float weights1[NUM_NODE][NUM_INPUT];
    float bias1[NUM_NODE];
    float weights2[NUM_NODE];
    float bias2;
};

static void fill_model(Model& m, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> w(-1.0f, 1.0f);
    std::uniform_real_distribution<float> s(0.5f, 1.5f);
    for (int i = 0; i < NUM_INPUT; ++i) {
        m.mean[i] = w(rng);
        m.istd[i] = 1.0f / s(rng);
    }
    for (int n = 0; n < NUM_NODE; ++n) {
        for (int i = 0; i < NUM_INPUT; ++i) m.weights1[n][i] = w(rng);
        m.bias1[n]    = w(rng);
        m.weights2[n] = w(rng);
    }
    m.bias2 = w(rng);
}

// ---------------------------------------------------------------------------
// CPU reference implementation — the source of truth (identical math to the
// baseline). The GPU result must match this to within the tolerance below.
// ---------------------------------------------------------------------------
static float evaluate_cpu(const Model& m, const float* x) {
    float in[NUM_INPUT];
    for (int i = 0; i < NUM_INPUT; ++i)
        in[i] = (x[i] - m.mean[i]) * m.istd[i];

    float h[NUM_NODE];
    for (int n = 0; n < NUM_NODE; ++n) {
        float acc = m.bias1[n];
        for (int i = 0; i < NUM_INPUT; ++i)
            acc += in[i] * m.weights1[n][i];
        h[n] = acc > 0.0f ? acc : 0.0f;
    }

    float out = m.bias2;
    for (int n = 0; n < NUM_NODE; ++n)
        out += h[n] * m.weights2[n];

    return 1.0f / (1.0f + std::exp(-out));
}

// ---------------------------------------------------------------------------
// OPTIMIZATION 2: model parameters live in constant memory. They are read by
// every thread and never change, so the constant cache broadcasts them.
// ---------------------------------------------------------------------------
__constant__ Model d_model_const;

// ---------------------------------------------------------------------------
// Optimized kernel.
//   - input is struct-of-arrays: input[i * num_events + idx] is feature i of
//     event idx, so threads idx and idx+1 read adjacent floats (coalesced).
//   - grid-stride loop covers all events with any launch configuration.
// ---------------------------------------------------------------------------
__global__ void infer_optimized(const float* __restrict__ input,  // [NUM_INPUT * num_events]
                                float* __restrict__ output,        // [num_events]
                                int num_events) {
    const Model& m = d_model_const;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_events;
         idx += blockDim.x * gridDim.x) {

        float in[NUM_INPUT];
        for (int i = 0; i < NUM_INPUT; ++i)
            in[i] = (input[(size_t)i * num_events + idx] - m.mean[i]) * m.istd[i];

        float h[NUM_NODE];
        for (int n = 0; n < NUM_NODE; ++n) {
            float acc = m.bias1[n];
            for (int i = 0; i < NUM_INPUT; ++i)
                acc += in[i] * m.weights1[n][i];
            h[n] = acc > 0.0f ? acc : 0.0f;
        }

        float out = m.bias2;
        for (int n = 0; n < NUM_NODE; ++n)
            out += h[n] * m.weights2[n];

        output[idx] = 1.0f / (1.0f + expf(-out));
    }
}

static void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    int num_events = (argc > 1) ? std::atoi(argv[1]) : 20'000'000;
    if (num_events <= 0) num_events = 20'000'000;

    std::printf("MVA inference (optimized)\n");
    std::printf("  events      : %d\n", num_events);
    std::printf("  input size  : %d\n", NUM_INPUT);
    std::printf("  hidden nodes: %d\n", NUM_NODE);

    Model model;
    fill_model(model, 12345u);

    // Host input in the SAME logical values as the baseline, but stored as
    // struct-of-arrays so the GPU reads are coalesced. h_aos keeps the
    // event-major copy the CPU reference expects.
    std::vector<float> h_aos((size_t)num_events * NUM_INPUT);   // event-major (for CPU ref)
    std::vector<float> h_soa((size_t)num_events * NUM_INPUT);   // feature-major (for GPU)
    {
        std::mt19937 rng(2024u);
        std::uniform_real_distribution<float> d(-3.0f, 3.0f);
        for (int e = 0; e < num_events; ++e)
            for (int i = 0; i < NUM_INPUT; ++i) {
                float v = d(rng);
                h_aos[(size_t)e * NUM_INPUT + i] = v;
                h_soa[(size_t)i * num_events + e] = v;
            }
    }
    std::vector<float> h_output(num_events);

    float* d_input = nullptr;
    float* d_output = nullptr;
    check_cuda(cudaMalloc(&d_input, h_soa.size() * sizeof(float)), "malloc input");
    check_cuda(cudaMalloc(&d_output, (size_t)num_events * sizeof(float)), "malloc output");

    check_cuda(cudaDeviceSynchronize(), "warmup sync");
    auto t0 = std::chrono::high_resolution_clock::now();

    check_cuda(cudaMemcpyToSymbol(d_model_const, &model, sizeof(Model)), "H2D model (const)");
    check_cuda(cudaMemcpy(d_input, h_soa.data(), h_soa.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "H2D input");

    int block = 256;
    int grid  = (num_events + block - 1) / block;
    if (grid > 65535) grid = 65535;                 // grid-stride handles the rest
    infer_optimized<<<grid, block>>>(d_input, d_output, num_events);
    check_cuda(cudaGetLastError(), "kernel launch");

    check_cuda(cudaMemcpy(h_output.data(), d_output, (size_t)num_events * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H output");
    check_cuda(cudaDeviceSynchronize(), "final sync");

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double throughput = num_events / secs;

    std::printf("  time (H2D+kernel+D2H): %.3f ms\n", secs * 1e3);
    std::printf("  throughput           : %.2f M events/sec\n", throughput / 1e6);

    // -------------------- correctness gate (same as baseline) --------------------
    // Sample events spread across the WHOLE range, always including the last
    // event, so a non-multiple-of-block count actually catches dropped bounds
    // checks / grid-stride off-by-one bugs (they corrupt the tail, not the head).
    const int sample = num_events < 10000 ? num_events : 10000;
    const int stride = num_events / sample;   // >= 1; spreads the sample over [0, num_events)
    double max_abs_err = 0.0;
    bool range_ok = true;
    auto check_event = [&](int i) {
        float ref = evaluate_cpu(model, &h_aos[(size_t)i * NUM_INPUT]);
        double err = std::fabs((double)ref - (double)h_output[i]);
        if (err > max_abs_err) max_abs_err = err;
        if (h_output[i] < 0.0f || h_output[i] > 1.0f) range_ok = false;
    };
    for (int s = 0; s < sample; ++s) check_event(s * stride);
    check_event(num_events - 1);   // always test the very last event (the tail)
    const double tol = 1e-5;
    bool pass = (max_abs_err <= tol) && range_ok;
    std::printf("  max abs error vs CPU : %.3e (tol %.1e)\n", max_abs_err, tol);
    std::printf("  sigmoid range [0,1]  : %s\n", range_ok ? "ok" : "VIOLATED");
    std::printf("  RESULT               : %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_input);
    cudaFree(d_output);
    return pass ? 0 : 1;
}
