#include "utils.cuh"

__global__ void CudaRandomInit(uint64_t seed, curandState *state) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}