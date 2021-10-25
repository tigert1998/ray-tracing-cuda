#include <tuple>

#include "ray_tracing.cuh"

using glm::vec3;
using std::pair;

constexpr int TRACE_DEPTH_LIMIT = 10;

__device__ glm::vec3 Trace(HitableList *world, Ray ray) {
  struct Layer {
    vec3 emitted;
    vec3 attenuation;
  };

  Layer storage[TRACE_DEPTH_LIMIT];

  int storage_size = 0;

  vec3 result;
  for (int depth = 0;; depth++) {
    HitRecord record;
    std::pair<double, double> t_range(1e-3, INFINITY);
    bool hit = world->Hit(ray, t_range, &record);
    if (!hit || depth >= TRACE_DEPTH_LIMIT) {
      result = vec3(0, 0, 0);
      break;
    }
    auto material_ptr = record.material_ptr;
    vec3 attenuation;
    Ray reflection;
    bool scattered =
        material_ptr->Scatter(ray, record, &attenuation, &reflection);
    auto hit_point = ray.position() + (float)record.t * ray.direction();
    auto emitted = material_ptr->Emit(0, 0, hit_point);
    if (!scattered) {
      result = emitted;
      break;
    }
    storage[storage_size].emitted = emitted;
    storage[storage_size++].attenuation = attenuation;
    ray = reflection;
  }

  for (int i = storage_size - 1; i >= 0; i--) {
    result = storage[i].emitted + storage[i].attenuation * result;
  }
  return result;
}