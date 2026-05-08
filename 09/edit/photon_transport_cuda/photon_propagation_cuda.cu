#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t status = (call);                                             \
        if (status != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(status)           \
                      << " at line " << __LINE__ << std::endl;                  \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (false)

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Absorber {
    Vec3 center;
    float radius;
    float density;
    float opacity;
};

__host__ __device__ inline Vec3 makeVec3(float x, float y, float z) {
    return Vec3{x, y, z};
}

__host__ __device__ inline Vec3 subtract(Vec3 a, Vec3 b) {
    return makeVec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline Vec3 add(Vec3 a, Vec3 b) {
    return makeVec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline Vec3 scale(Vec3 value, float factor) {
    return makeVec3(value.x * factor, value.y * factor, value.z * factor);
}

__host__ __device__ inline float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float length(Vec3 value) {
    return sqrtf(dot(value, value));
}

__host__ __device__ inline Vec3 normalize(Vec3 value) {
    const float norm = fmaxf(length(value), 1.0e-12f);
    return scale(value, 1.0f / norm);
}

// Compute how much of the segment from start to end lies inside one spherical cloud.
__device__ float segmentSpherePathLength(Vec3 start, Vec3 end, const Absorber& sphere) {
    const Vec3 direction = subtract(end, start);
    const float segmentLength = length(direction);
    if (segmentLength < 1.0e-12f) {
        return 0.0f;
    }

    const Vec3 rayDir = scale(direction, 1.0f / segmentLength);
    const Vec3 oc = subtract(start, sphere.center);
    const float b = 2.0f * dot(oc, rayDir);
    const float c = dot(oc, oc) - sphere.radius * sphere.radius;
    const float discriminant = b * b - 4.0f * c;
    if (discriminant <= 0.0f) {
        return 0.0f;
    }

    const float sqrtDisc = sqrtf(discriminant);
    const float t0 = (-b - sqrtDisc) * 0.5f;
    const float t1 = (-b + sqrtDisc) * 0.5f;
    const float tEnter = fmaxf(0.0f, fminf(t0, t1));
    const float tExit = fminf(segmentLength, fmaxf(t0, t1));
    if (tExit <= tEnter) {
        return 0.0f;
    }
    return tExit - tEnter;
}

__host__ __device__ inline float toyRedshift(float distanceMpc, float hubbleDistanceMpc = 4300.0f) {
    return distanceMpc / hubbleDistanceMpc;
}

__host__ __device__ inline float observedFlux(float luminosity, float distanceMpc, float tau) {
    const float z = toyRedshift(distanceMpc);
    const float luminosityDistance = distanceMpc * (1.0f + z);
    return luminosity * expf(-tau) /
           (4.0f * 3.14159265358979323846f * luminosityDistance * luminosityDistance);
}

// One CUDA thread handles one source galaxy or one detector line of sight.
__global__ void propagatePhotonsKernel(
    const Vec3* sourcePositions,
    const float* sourceLuminosities,
    const Absorber* absorbers,
    int numSources,
    int numAbsorbers,
    Vec3 observer,
    float* opticalDepths,
    float* redshifts,
    float* fluxes) {
    const int sourceIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (sourceIndex >= numSources) {
        return;
    }

    const Vec3 sourcePosition = sourcePositions[sourceIndex];
    const float luminosity = sourceLuminosities[sourceIndex];
    const float distanceMpc = length(subtract(sourcePosition, observer));

    float tau = 0.0f;
    for (int absorberIndex = 0; absorberIndex < numAbsorbers; ++absorberIndex) {
        const Absorber absorber = absorbers[absorberIndex];
        const float pathLength = segmentSpherePathLength(observer, sourcePosition, absorber);
        tau += absorber.opacity * absorber.density * pathLength;
    }

    opticalDepths[sourceIndex] = tau;
    redshifts[sourceIndex] = toyRedshift(distanceMpc);
    fluxes[sourceIndex] = observedFlux(luminosity, distanceMpc, tau);
}

std::vector<Vec3> buildSourceShell(int numSources, float shellRadiusMpc) {
    std::vector<Vec3> sources(numSources);
    for (int index = 0; index < numSources; ++index) {
        const float u = (index + 0.5f) / static_cast<float>(numSources);
        const float v = (index * 0.61803398875f) - floorf(index * 0.61803398875f);
        const float phi = 2.0f * 3.14159265358979323846f * v;
        const float cosTheta = 1.0f - 2.0f * u;
        const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
        const Vec3 direction = makeVec3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
        sources[index] = scale(direction, shellRadiusMpc);
    }
    return sources;
}

std::vector<float> buildLuminosities(int numSources) {
    std::vector<float> luminosities(numSources);
    for (int index = 0; index < numSources; ++index) {
        luminosities[index] = 0.85f + 0.30f * ((index % 17) / 16.0f);
    }
    return luminosities;
}

std::vector<Absorber> buildAbsorbers(int numAbsorbers) {
    std::vector<Absorber> absorbers(numAbsorbers);
    for (int index = 0; index < numAbsorbers; ++index) {
        const float phase = 0.37f * static_cast<float>(index + 1);
        absorbers[index] = Absorber{
            makeVec3(420.0f * cosf(phase), 380.0f * sinf(1.7f * phase), -140.0f - 18.0f * index),
            35.0f + 3.0f * static_cast<float>(index % 11),
            0.006f + 0.0012f * static_cast<float>(index % 7),
            0.7f + 0.08f * static_cast<float>(index % 5)};
    }
    return absorbers;
}

int main() {
    constexpr int numSources = 4096;
    constexpr int numAbsorbers = 64;
    constexpr float sourceShellMpc = 1200.0f;
    const Vec3 observer = makeVec3(0.0f, 0.0f, 0.0f);

    const std::vector<Vec3> hostSources = buildSourceShell(numSources, sourceShellMpc);
    const std::vector<float> hostLuminosities = buildLuminosities(numSources);
    const std::vector<Absorber> hostAbsorbers = buildAbsorbers(numAbsorbers);

    Vec3* sourcePositions = nullptr;
    float* sourceLuminosities = nullptr;
    Absorber* absorbers = nullptr;
    float* opticalDepths = nullptr;
    float* redshifts = nullptr;
    float* fluxes = nullptr;

    // Unified memory keeps the example short and readable for teaching.
    CUDA_CHECK(cudaMallocManaged(&sourcePositions, numSources * sizeof(Vec3)));
    CUDA_CHECK(cudaMallocManaged(&sourceLuminosities, numSources * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&absorbers, numAbsorbers * sizeof(Absorber)));
    CUDA_CHECK(cudaMallocManaged(&opticalDepths, numSources * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&redshifts, numSources * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&fluxes, numSources * sizeof(float)));

    std::copy(hostSources.begin(), hostSources.end(), sourcePositions);
    std::copy(hostLuminosities.begin(), hostLuminosities.end(), sourceLuminosities);
    std::copy(hostAbsorbers.begin(), hostAbsorbers.end(), absorbers);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (numSources + threadsPerBlock - 1) / threadsPerBlock;

    propagatePhotonsKernel<<<blocksPerGrid, threadsPerBlock>>>(
        sourcePositions,
        sourceLuminosities,
        absorbers,
        numSources,
        numAbsorbers,
        observer,
        opticalDepths,
        redshifts,
        fluxes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float meanTau = 0.0f;
    float maxTau = 0.0f;
    float meanFlux = 0.0f;
    for (int index = 0; index < numSources; ++index) {
        meanTau += opticalDepths[index];
        meanFlux += fluxes[index];
        maxTau = fmaxf(maxTau, opticalDepths[index]);
    }
    meanTau /= static_cast<float>(numSources);
    meanFlux /= static_cast<float>(numSources);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Toy CUDA photon propagation" << std::endl;
    std::cout << "Sources:          " << numSources << std::endl;
    std::cout << "Absorbers:        " << numAbsorbers << std::endl;
    std::cout << "Launch:           " << blocksPerGrid << " blocks x " << threadsPerBlock
              << " threads" << std::endl;
    std::cout << "Mean optical tau: " << meanTau << std::endl;
    std::cout << "Max optical tau:  " << maxTau << std::endl;
    std::cout << "Mean flux:        " << meanFlux << std::endl;
    std::cout << "Sample source 0:  z=" << redshifts[0] << " flux=" << fluxes[0] << std::endl;
    std::cout << "Sample source 1:  z=" << redshifts[1] << " flux=" << fluxes[1] << std::endl;

    CUDA_CHECK(cudaFree(sourcePositions));
    CUDA_CHECK(cudaFree(sourceLuminosities));
    CUDA_CHECK(cudaFree(absorbers));
    CUDA_CHECK(cudaFree(opticalDepths));
    CUDA_CHECK(cudaFree(redshifts));
    CUDA_CHECK(cudaFree(fluxes));
    return 0;
}
