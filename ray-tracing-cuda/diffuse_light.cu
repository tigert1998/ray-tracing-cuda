#include "diffuse_light.cuh"

using namespace glm;

__device__ bool DiffuseLight::Scatter(const Ray &ray, const HitRecord &record,
                                      curandState *state, glm::vec3 *out_albedo,
                                      Ray *out_ray) {
  return false;
}

__device__ vec3 DiffuseLight::Emit(double u, double v, const vec3 &p) const {
  return texture_ptr_->Value(u, v, p);
}

__host__ __device__ DiffuseLight::DiffuseLight(Texture *texture_ptr) {
  texture_ptr_ = texture_ptr;
}