#pragma once

class Material;

#include <glm/glm.hpp>

#include "cuda_copyable.cuh"
#include "hitable.cuh"
#include "ray.cuh"

class Material : public CudaCopyable {
 public:
  __device__ virtual bool Scatter(const Ray &ray, const HitRecord &record,
                                  std::pair<glm::vec3, Ray> *out) = 0;
  __device__ virtual glm::vec3 Emit(double u, double v,
                                    const glm::vec3 &p) const;
};
