#include "triangle.cuh"
#include "utils.cuh"

using namespace glm;

__host__ __device__ Triangle::Triangle(glm::vec3 p[], Material *material_ptr)
    : material_ptr_(material_ptr) {
  memcpy(p_, p, sizeof(glm::vec3) * 3);
}

__device__ bool Triangle::Hit(const Ray &ray, double t_from, double t_to,
                              HitRecord *out) {
  if (TriangleHit(p_, ray, t_from, t_to, &out->t, &out->normal, &out->u,
                  &out->v)) {
    out->material_ptr = material_ptr_;
    return true;
  }
  return false;
}