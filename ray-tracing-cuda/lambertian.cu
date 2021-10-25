#include "lambertian.cuh"
#include "utils.cuh"

using std::pair;
using namespace glm;

Lambertian::Lambertian(curandState *state, Texture *texture_ptr) {
  state_ = state;
  texture_ptr_ = texture_ptr;
}

glm::vec3 Lambertian::SphericalRand() {
  float x, y, z, l;
  do {
    x = cudaRandomFloat(-1, 1, state_);
    y = cudaRandomFloat(-1, 1, state_);
    z = cudaRandomFloat(-1, 1, state_);
    l = pow(x * x + y * y + z * z, 0.5);
  } while (l > 1);
  x /= l;
  y /= l;
  z /= l;
  return vec3(x, y, z);
}

bool Lambertian::Scatter(const Ray &ray, const HitRecord &record,
                         std::pair<glm::vec3, Ray> *out) {
  if (dot(ray.direction(), record.normal) >= 0) return false;
  auto p = ray.position() + float(record.t) * ray.direction();
  auto albedo = texture_ptr_->Value(0, 0, p);
  vec3 next_dir = normalize(SphericalRand() + record.normal);
  *out = pair<vec3, Ray>(albedo, Ray(p, next_dir));
  return true;
}
