#pragma once

#include <glm/glm.hpp>

#include "hitable.cuh"

class Triangle : public Hitable {
 private:
  glm::vec3 p_[3];
  Material *material_ptr_;

 public:
  Triangle() = delete;
  __host__ __device__ Triangle(glm::vec3 p[], Material *material_ptr);
  __device__ bool Hit(const Ray &ray, std::pair<double, double> t_range,
                      HitRecord *out) override;
};