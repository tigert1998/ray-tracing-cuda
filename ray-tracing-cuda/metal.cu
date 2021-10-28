#include "glm/geometric.hpp"
#include "metal.cuh"

using namespace glm;

__device__ Metal::Metal(vec3 albedo) : albedo_(albedo) {}

__device__ bool Metal::Scatter(const Ray &ray, const HitRecord &record,
                               curandState *state, glm::vec3 *out_albedo,
                               Ray *out_ray) {
  if (dot(ray.direction(), record.normal) >= 0) return false;
  vec3 p = ray.position() + float(record.t) * ray.direction();
  *out_albedo = albedo_;
  *out_ray = Ray(p, reflect(ray.direction(), record.normal));
  return true;
}