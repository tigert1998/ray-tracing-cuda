#include <tuple>

#include "lambertian.cuh"
#include "utils.cuh"

using std::pair;
using namespace glm;

__host__ __device__ Lambertian::Lambertian(Texture *texture_ptr) {
  texture_ptr_ = texture_ptr;
  use_constant_tex_ = false;
}

__host__ __device__ Lambertian::Lambertian(glm::vec3 color) {
  color_ = ConstantTexture(color);
  use_constant_tex_ = true;
}

__device__ glm::vec3 Lambertian::SphericalRand(curandState *state) {
  float x, y, z, l;
  do {
    x = CudaRandomFloat(-1, 1, state);
    y = CudaRandomFloat(-1, 1, state);
    z = CudaRandomFloat(-1, 1, state);
    l = pow(x * x + y * y + z * z, 0.5);
  } while (l > 1);
  x /= l;
  y /= l;
  z /= l;
  return vec3(x, y, z);
}

__device__ bool Lambertian::Scatter(const Ray &ray, const HitRecord &record,
                                    curandState *state, glm::vec3 *out_albedo,
                                    Ray *out_ray) {
  if (dot(ray.direction(), record.normal) >= 0) return false;
  auto p = ray.position() + float(record.t) * ray.direction();
  auto albedo = texture_ptr()->Value(0, 0, p);
  vec3 next_dir = normalize(SphericalRand(state) + record.normal);
  *out_albedo = albedo;
  *out_ray = Ray(p, next_dir);
  return true;
}
