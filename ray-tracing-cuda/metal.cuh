#pragma once

#include "material.cuh"

class Metal : public Material {
 public:
  Metal() = delete;
  __device__ Metal(glm::vec3 albedo);
  __device__ bool Scatter(const Ray &ray, const HitRecord &record,
                          curandState *state, glm::vec3 *out_albedo,
                          Ray *out_ray);

 private:
  glm::vec3 albedo_;
};