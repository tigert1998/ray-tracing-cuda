#include <tuple>

#include "lambertian.cuh"
#include "utils.cuh"

using std::pair;
using namespace glm;

Lambertian::Lambertian(curandState *state, Texture *texture_ptr) {
  state_ = state;
  texture_ptr_ = texture_ptr;
}

__device__ glm::vec3 Lambertian::SphericalRand() {
  float x, y, z, l;
  do {
    x = CudaRandomFloat(-1, 1, state_);
    y = CudaRandomFloat(-1, 1, state_);
    z = CudaRandomFloat(-1, 1, state_);
    l = pow(x * x + y * y + z * z, 0.5);
  } while (l > 1);
  x /= l;
  y /= l;
  z /= l;
  return vec3(x, y, z);
}

__device__ bool Lambertian::Scatter(const Ray &ray, const HitRecord &record,
                                    glm::vec3 *out_albedo, Ray *out_ray) {
  if (dot(ray.direction(), record.normal) >= 0) return false;
  auto p = ray.position() + float(record.t) * ray.direction();
  auto albedo = texture_ptr_->Value(0, 0, p);
  vec3 next_dir = normalize(SphericalRand() + record.normal);
  *out_albedo = albedo;
  *out_ray = Ray(p, next_dir);
  return true;
}
