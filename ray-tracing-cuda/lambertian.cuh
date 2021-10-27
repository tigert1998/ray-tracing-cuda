#pragma once

#include <curand_kernel.h>

#include <glm/glm.hpp>

#include "material.cuh"
#include "textures/constant_texture.cuh"
#include "textures/texture.cuh"

class Lambertian : public Material {
 public:
  __host__ __device__ Lambertian(glm::vec3 color);
  __host__ __device__ Lambertian(Texture *texture_ptr);
  __device__ bool Scatter(const Ray &ray, const HitRecord &record,
                          curandState *state, glm::vec3 *out_albedo,
                          Ray *out_ray) override;

 private:
  Texture *texture_ptr_;
  ConstantTexture color_;
  bool use_constant_tex_;

  __device__ glm::vec3 SphericalRand(curandState *state);

  __device__ __host__ __inline__ Texture *texture_ptr() {
    return use_constant_tex_ ? &color_ : texture_ptr_;
  }
};
