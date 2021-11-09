#include "parallelogram.cuh"
#include "triangle.cuh"
#include "utils.cuh"

using glm::cross;
using glm::dot;
using glm::normalize;
using glm::vec3;

__host__ __device__ Parallelogram::Parallelogram(vec3 p[3],
                                                 Material *material_ptr) {
  for (int i = 0; i <= 1; i++) p_[i] = p[i];
  p_[2] = p[1] + p[2] - p[0];
  p_[3] = p[2];
  p_[4] = p[0];
  material_ptr_ = material_ptr;
}

__device__ bool Parallelogram::Hit(const Ray &ray, double t_from, double t_to,
                                   HitRecord *out) {
  HitRecord record;
  if (TriangleHit(p_, ray, t_from, t_to, &record.t, &record.normal) ||
      TriangleHit(p_ + 2, ray, t_from, t_to, &record.t, &record.normal)) {
    record.material_ptr = material_ptr_;
    *out = record;
    return true;
  }

  return false;
}