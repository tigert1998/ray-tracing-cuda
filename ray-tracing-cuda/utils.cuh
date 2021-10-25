#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

__inline__ __device__ float cudaRandomFloat(float min, float max,
                                            curandState *state) {
  // (min, max]
  float t = curand_uniform(state);
  return t * (max - min) + min;
}