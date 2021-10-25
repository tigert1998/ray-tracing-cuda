#pragma once

#include <glm/glm.hpp>

#include "cuda_copyable.cuh"

class Texture : public CudaCopyable {
 public:
  __device__ virtual glm::vec3 Value(double u, double v,
                                     const glm::vec3 &p) const = 0;
};