cmake_minimum_required(VERSION 3.21)
project(photon_propagation_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(photon_propagation_cuda photon_propagation_cuda.cu)
target_link_libraries(photon_propagation_cuda PRIVATE CUDA::cudart)
