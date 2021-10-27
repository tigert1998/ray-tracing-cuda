#pragma once

struct HitRecord;

#include <cuda_runtime.h>

#include <memory>

#include "cuda_copyable.cuh"
#include "material.cuh"
#include "ray.cuh"

struct HitRecord {
  double t;
  glm::vec3 normal;
  Material *material_ptr;
};

class Hitable : public CudaCopyable {
 public:
  __device__ virtual bool Hit(const Ray &ray, double t_from, double t_to,
                              HitRecord *out) = 0;
};
