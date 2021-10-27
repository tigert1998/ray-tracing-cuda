#pragma once

class Material;

#include <curand_kernel.h>

#include <glm/glm.hpp>

#include "cuda_copyable.cuh"
#include "hitable.cuh"
#include "ray.cuh"

class Material : public CudaCopyable {
 public:
  __device__ virtual bool Scatter(const Ray &ray, const HitRecord &record,
                                  curandState *state, glm::vec3 *out_albedo,
                                  Ray *out_ray) = 0;
  __device__ virtual glm::vec3 Emit(double u, double v,
                                    const glm::vec3 &p) const;
};
