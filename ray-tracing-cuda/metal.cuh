#pragma once

#include "material.cuh"

class Metal : public Material {
 public:
  Metal() = delete;
  __device__ Metal(glm::vec3 albedo);
  __device__ Metal(glm::vec3 albedo, float fuzz);
  __device__ bool Scatter(const Ray &ray, const HitRecord &record,
                          curandState *state, glm::vec3 *out_albedo,
                          Ray *out_ray);

 private:
  glm::vec3 albedo_;
  float fuzz_;

  __device__ glm::vec3 RandomInUnitSphere(curandState *state);
};