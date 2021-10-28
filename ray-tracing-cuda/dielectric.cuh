#pragma once

#include "material.cuh"

class Dielectric : public Material {
 public:
  Dielectric() = delete;
  __device__ __host__ Dielectric(glm::vec3 attenuation,
                                 double refractive_index);
  __device__ bool Scatter(const Ray &ray, const HitRecord &record,
                          curandState *state, glm::vec3 *out_albedo,
                          Ray *out_ray);

 private:
  glm::vec3 attenuation_;
  double refractive_index_;
};