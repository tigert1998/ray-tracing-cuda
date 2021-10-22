#include "sphere.cuh"

using namespace glm;

Sphere::Sphere(vec3 position, double radius, Material* material_ptr)
    : radius_(radius), position_(position), material_ptr_(material_ptr) {}

__device__ bool Sphere::Hit(const Ray& ray, std::pair<double, double> t_range,
                 HitRecord* out) const {
  double a = pow(length(ray.direction()), 2);
  double b = 2 * dot(ray.direction(), ray.position() - this->position());
  double c = pow(length(ray.position() - this->position()), 2) -
             pow(this->radius(), 2);
  double discriminant = pow(b, 2) - 4 * a * c;
  if (discriminant < 0) {
    return false;
  }
  double t = (-b - pow(discriminant, 0.5)) / (2 * a);
  if (t_range.first <= t && t <= t_range.second) {
    HitRecord record;
    record.t = t;
    vec3 p = ray.position() + float(t) * ray.direction();
    record.normal = normalize(p - this->position());
    record.material_ptr = material_ptr();
    *out = record;
    return true;
  }
  t = (-b + pow(discriminant, 0.5)) / (2 * a);
  if (t_range.first <= t && t <= t_range.second) {
    HitRecord record;
    record.t = t;
    vec3 p = ray.position() + float(t) * ray.direction();
    record.normal = normalize(p - this->position());
    record.material_ptr = material_ptr();
    *out = record;
    return true;
  }
  return false;
}

double Sphere::radius() const { return radius_; }

vec3 Sphere::position() const { return position_; }

Material* Sphere::material_ptr() const { return material_ptr_; }
