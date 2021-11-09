#include "glm/geometric.hpp"
#include "metal.cuh"
#include "utils.cuh"

using namespace glm;

__device__ Metal::Metal(vec3 albedo) : Metal(albedo, 0) {}

__device__ Metal::Metal(glm::vec3 albedo, float fuzz)
    : albedo_(albedo), fuzz_(fuzz < 1 ? fuzz : 1) {}

__device__ bool Metal::Scatter(const Ray &ray, const HitRecord &record,
                               curandState *state, glm::vec3 *out_albedo,
                               Ray *out_ray) {
  if (dot(ray.direction(), record.normal) >= 0) return false;
  vec3 p = ray.position() + float(record.t) * ray.direction();
  *out_albedo = albedo_;
  auto reflected = reflect(ray.direction(), record.normal);
  if (fuzz_ > 0) {
    *out_ray = Ray(p, reflected + fuzz_ * RandomInUnitSphere(state));
  } else {
    *out_ray = Ray(p, reflected);
  }
  return true;
}

__device__ glm::vec3 Metal::RandomInUnitSphere(curandState *state) {
  float x, y, z, l;
  do {
    x = CudaRandomFloat(-1, 1, state);
    y = CudaRandomFloat(-1, 1, state);
    z = CudaRandomFloat(-1, 1, state);
    l = pow(x * x + y * y + z * z, 0.5);
  } while (l > 1);
  return vec3(x, y, z);
}