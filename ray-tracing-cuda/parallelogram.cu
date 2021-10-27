#include "parallelogram.cuh"
#include "triangle.cuh"

using glm::cross;
using glm::dot;
using glm::normalize;
using glm::vec3;

__host__ __device__ Parallelogram::Parallelogram(vec3 p[3],
                                                 Material *material_ptr) {
  for (int i = 0; i <= 1; i++) p_[i] = p[i];
  p_[2] = p[1] + p[2] - p[0];
  p_[3] = p[2];
  material_ptr_ = material_ptr;
}

__device__ bool Parallelogram::Hit(const Ray &ray, double t_from, double t_to,
                                   HitRecord *out) {
  vec3 n = normalize(cross(p_[2] - p_[0], p_[1] - p_[2]));
  double den = dot(ray.direction(), n);
  double num = dot(p_[0] - ray.position(), n);
  double t = num / den;
  if (std::isnan(t) || std::isinf(t)) return false;
  if (t < t_from || t > t_to) return false;
  vec3 hit_point = ray.position() + (float)t * ray.direction();
  {
    double dot_values[4];
    for (int i = 0; i < 4; i++) {
      vec3 a = p_[i], b = p_[(i + 1) % 4];
      dot_values[i] = dot(cross(a - hit_point, b - hit_point), n);
    }
    for (int i = 0; i < 4; i++)
      if (dot_values[i] * dot_values[(i + 1) % 4] < 0) return false;
  }
  HitRecord record;
  record.t = t;
  record.normal = dot(n, ray.direction()) < 0 ? n : -n;
  record.material_ptr = material_ptr_;
  *out = record;
  return true;
}