#pragma once

#include <utility>

#include "hitable.cuh"
#include "material.cuh"

class SkyMaterial : public Material {
 public:
  __device__ bool Scatter(const Ray &ray, const HitRecord &record,
                          glm::vec3 *out_albedo, Ray *out_ray) override;
  __device__ glm::vec3 Emit(double u, double v,
                            const glm::vec3 &p) const override;
};

class Sky : public Hitable {
 public:
  Sky();
  __device__ bool Hit(const Ray &ray, std::pair<double, double> t_range,
                      HitRecord *out);
  __device__ __host__ Material *material_ptr();

 private:
  SkyMaterial material_;
};