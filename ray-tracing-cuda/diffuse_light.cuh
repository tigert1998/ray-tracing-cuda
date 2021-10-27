#pragma once

#include "material.cuh"
#include "textures/texture.cuh"

class DiffuseLight : public Material {
 private:
  Texture *texture_ptr_;

 public:
  DiffuseLight() = delete;
  __host__ __device__ DiffuseLight(Texture *texture_ptr);
  __device__ bool Scatter(const Ray &ray, const HitRecord &record,
                          curandState *state, glm::vec3 *out_albedo,
                          Ray *out_ray) override;
  __device__ glm::vec3 Emit(double u, double v,
                            const glm::vec3 &p) const override;
};