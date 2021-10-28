#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

#include <glm/glm.hpp>
#include <nvfunctional>
#include <string>
#include <vector>

#include "ray.cuh"

__inline__ __device__ float CudaRandomFloat(float min, float max,
                                            curandState *state) {
  // (min, max]
  float t = curand_uniform(state);
  return t * (max - min) + min;
}

__global__ void CudaRandomInit(uint64_t seed, curandState *state);

void WriteImage(std::vector<glm::vec3> &pixels, int height, int width,
                const std::string &path);

__device__ bool TriangleHit(glm::vec3 p[3], const Ray &ray, double t_from,
                            double t_to, double *out_t, glm::vec3 *out_normal);

template <typename T>
__device__ void QuickSort(
    T *array, int low, int high,
    const nvstd::function<int(const T &, const T &)> &comp) {
  int i = low;
  int j = high;
  T &pivot = array[(i + j) / 2];

  while (i <= j) {
    while (comp(array[i], pivot) < 0) i++;
    while (comp(array[j], pivot) > 0) j--;
    if (i <= j) {
      auto temp = array[i];
      array[i] = array[j];
      array[j] = temp;
      i++;
      j--;
    }
  }
  if (j > low) QuickSort(array, low, j, comp);
  if (i < high) QuickSort(array, i, high, comp);
}
