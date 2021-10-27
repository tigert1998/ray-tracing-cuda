#pragma once

#include <utility>

#include "hitable.cuh"
#include "material.cuh"

class Sphere : public Hitable {
 public:
  Sphere() = delete;
  __host__ __device__ Sphere(glm::vec3 position, double radius,
                             Material* material_ptr);
  __device__ bool Hit(const Ray& ray, double t_from, double t_to,
                      HitRecord* out);
  __device__ __host__ double radius() const;
  __device__ __host__ glm::vec3 position() const;
  __device__ __host__ Material* material_ptr() const;

 private:
  double radius_;
  glm::vec3 position_;
  Material* material_ptr_;
};
