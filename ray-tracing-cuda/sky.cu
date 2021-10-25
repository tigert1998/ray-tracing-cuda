#include "sky.cuh"

__device__ bool SkyMaterial::Scatter(const Ray &ray, const HitRecord &record,
                                     glm::vec3 *out_albedo, Ray *out_ray) {
  return false;
}

__device__ glm::vec3 SkyMaterial::Emit(double u, double v,
                                       const glm::vec3 &p) const {
  glm::vec3 dir = glm::normalize(p);
  float t = 0.5 * (dir.y + 1.0);
  return (1.0f - t) * glm::vec3(1.0, 1.0, 1.0) + t * glm::vec3(0.5, 0.7, 1.0);
}

__host__ __device__ Sky::Sky() {}

__device__ bool Sky::Hit(const Ray &ray, std::pair<double, double> t_range,
                         HitRecord *out) {
  double t = 1e9;
  if (t_range.first <= t && t <= t_range.second) {
    out->material_ptr = &material_;
    out->t = t;
    return true;
  }
  return false;
}

__device__ __host__ Material *Sky::material_ptr() { return &material_; }
