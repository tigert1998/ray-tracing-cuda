#include "parallelogram.cuh"
#include "triangle.cuh"
#include "utils.cuh"

using glm::cross;
using glm::dot;
using glm::normalize;
using glm::vec3;

__host__ __device__ Parallelogram::Parallelogram(vec3 p[3],
                                                 Material *material_ptr) {
  for (int i = 0; i <= 2; i++) p_[i] = p[i];
  p_[3] = p[1] + p[2] - p[0];
  material_ptr_ = material_ptr;
}

__device__ bool Parallelogram::Hit(const Ray &ray, double t_from, double t_to,
                                   HitRecord *out) {
  // <0, 1> 0 - 1 <1, 1>
  //        | / |
  // <0, 0> 2 - 3 <1, 0>
  HitRecord record;
  record.material_ptr = material_ptr_;
  double u, v;
  if (TriangleHit(p_, ray, t_from, t_to, &record.t, &record.normal, &u, &v)) {
    glm::vec2 uv = glm::vec2(0, 1) * (float)(1 - u - v) +
                   glm::vec2(1, 1) * (float)u + glm::vec2(0, 0) * (float)v;
    record.u = uv.x;
    record.v = uv.y;
    *out = record;
    return true;
  }
  if (TriangleHit(p_ + 1, ray, t_from, t_to, &record.t, &record.normal, &u,
                  &v)) {
    glm::vec2 uv = glm::vec2(1, 1) * (float)(1 - u - v) +
                   glm::vec2(0, 0) * (float)u + glm::vec2(1, 0) * (float)v;
    record.u = uv.x;
    record.v = uv.y;
    *out = record;
    return true;
  }

  return false;
}