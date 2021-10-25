#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

__inline__ __device__ float CudaRandomFloat(float min, float max,
                                            curandState *state) {
  // (min, max]
  float t = curand_uniform(state);
  return t * (max - min) + min;
}

__global__ void CudaRandomInit(uint64_t seed, curandState *state);