#pragma once

#include <glm/glm.hpp>

#include "hitable_list.cuh"

class Parallelogram : public Hitable {
 private:
  glm::vec3 p_[4];
  Material *material_ptr_;

 public:
  Parallelogram() = delete;
  __host__ __device__ Parallelogram(glm::vec3 p[3], Material *material_ptr);
  __device__ bool Hit(const Ray &ray, std::pair<double, double> t_range,
                      HitRecord *out) override;
};