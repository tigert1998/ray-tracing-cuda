#pragma once

struct HitRecord;

#include <cuda_runtime.h>

#include <memory>

#include "material.cuh"
#include "ray.cuh"

struct HitRecord {
  double t;
  glm::vec3 normal;
  Material *material_ptr;
};

class Hitable {
 public:
  __device__ virtual bool Hit(const Ray &ray, std::pair<double, double> t_range,
                              HitRecord *out) const = 0;
};
