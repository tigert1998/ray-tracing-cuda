#pragma once

#include <utility>

#include "hitable.cuh"
#include "material.cuh"

class SkyMaterial : public Material {
 public:
  __device__ bool Scatter(const Ray &ray, const HitRecord &record,
                          curandState *, glm::vec3 *out_albedo,
                          Ray *out_ray) override;
  __device__ glm::vec3 Emit(double u, double v,
                            const glm::vec3 &p) const override;
};

class Sky : public Hitable {
 public:
  __host__ __device__ Sky();
  __device__ bool Hit(const Ray &ray, double t_from, double t_to,
                      HitRecord *out);
  __device__ __host__ Material *material_ptr();

 private:
  SkyMaterial material_;
};
