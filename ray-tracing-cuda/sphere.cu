#include <glm/gtc/constants.hpp>

#include "sphere.cuh"

using namespace glm;

__host__ __device__ Sphere::Sphere(vec3 position, double radius,
                                   Material* material_ptr)
    : radius_(radius), position_(position), material_ptr_(material_ptr) {}

__device__ bool Sphere::Hit(const Ray& ray, double t_from, double t_to,
                            HitRecord* out) {
  double a = pow(length(ray.direction()), 2);
  double b = 2 * dot(ray.direction(), ray.position() - this->position());
  double c = pow(length(ray.position() - this->position()), 2) -
             pow(this->radius(), 2);
  double discriminant = pow(b, 2) - 4 * a * c;
  if (discriminant < 0) {
    return false;
  }
  double t = (-b - pow(discriminant, 0.5)) / (2 * a);
  if (t_from <= t && t <= t_to) {
    HitRecord record;
    record.t = t;
    vec3 p = ray.position() + float(t) * ray.direction();
    record.normal = normalize(p - this->position());
    record.material_ptr = material_ptr();
    GetUV(record.normal, &record.u, &record.v);
    *out = record;
    return true;
  }
  t = (-b + pow(discriminant, 0.5)) / (2 * a);
  if (t_from <= t && t <= t_to) {
    HitRecord record;
    record.t = t;
    vec3 p = ray.position() + float(t) * ray.direction();
    record.normal = normalize(p - this->position());
    record.material_ptr = material_ptr();
    GetUV(record.normal, &record.u, &record.v);
    *out = record;
    return true;
  }
  return false;
}

double Sphere::radius() const { return radius_; }

vec3 Sphere::position() const { return position_; }

Material* Sphere::material_ptr() const { return material_ptr_; }

__device__ void Sphere::GetUV(const glm::vec3& p, double* u, double* v) {
  // p: a given point on the sphere of radius one, centered at the origin.
  // u: returned value [0,1] of angle around the Y axis from X=-1.
  // v: returned value [0,1] of angle from Y=-1 to Y=+1.
  //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
  //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
  //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

  auto theta = acos(-p.y);
  auto phi = atan2(-p.z, p.x) + pi<float>();
  *u = phi / (2 * pi<float>());
  *v = theta / pi<float>();
}