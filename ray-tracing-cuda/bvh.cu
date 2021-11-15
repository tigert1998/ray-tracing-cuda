#include "bvh.cuh"
#include "utils.cuh"

using glm::vec3;

__device__ bool AABB::CheckOnPlane(const Ray &ray, double t, double t_from,
                                   double t_to, int axis) {
  if (std::isnan(t) || std::isinf(t)) return false;
  if (!(t_from <= t && t <= t_to)) return false;
  vec3 pt_on_plane = ray.position() + (float)t * ray.direction();
  for (int i = 0; i < 3; i++) {
    if (i == axis) continue;
    if (min[i] <= pt_on_plane[i] && pt_on_plane[i] <= max[i]) continue;
    return false;
  }
  return true;
}

__device__ bool AABB::Hit(const Ray &ray, double t_from, double t_to,
                          HitRecord *out) {
  vec3 dir = ray.direction(), pos = ray.position();

  for (int i = 0; i < 3; i++) {
    if (dir[i] == 0.f) continue;
    double ts[] = {(min[i] - pos[i]) / dir[i], (max[i] - pos[i]) / dir[i]};
    if (CheckOnPlane(ray, ts[0], t_from, t_to, i)) return true;
    if (CheckOnPlane(ray, ts[1], t_from, t_to, i)) return true;
  }
  return false;
}
