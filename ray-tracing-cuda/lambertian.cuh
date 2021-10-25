#pragma once

#include <curand_kernel.h>

#include "material.cuh"
#include "textures/texture.cuh"

class Lambertian : public Material {
 public:
  Lambertian(curandState *state, Texture *texture_ptr);
  __device__ bool Scatter(const Ray &ray, const HitRecord &record,
                          glm::vec3 *out_albedo, Ray *out_ray) override;

 private:
  curandState *state_;
  Texture *texture_ptr_;

  __device__ glm::vec3 SphericalRand();
};
