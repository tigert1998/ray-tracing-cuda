#include <curand_kernel.h>

#include <tuple>

#include "ray_tracing.cuh"
#include "utils.cuh"

using glm::vec3;
using std::pair;

constexpr int TRACE_DEPTH_LIMIT = 10;

__device__ glm::vec3 Trace(HitableList *world, Ray ray, curandState *states) {
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
        material_ptr->Scatter(ray, record, states, &attenuation, &reflection);
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

__global__ void RayTracing(HitableList *world, Camera *camera, int height,
                           int width, int spp, curandState *states,
                           vec3 *out_image) {
  // 0 <= i < height
  // 0 <= j < width
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= height || j >= width) return;
  int idx = i * width + j;

  auto color = vec3(0);
  for (int k = 0; k < spp; k++) {
    double x =
        (CudaRandomFloat(0, 1, states + idx) + double(j)) / double(width);
    double y = (CudaRandomFloat(0, 1, states + idx) + double(height - i)) /
               double(height);
    x = 2 * x - 1;
    y = 2 * y - 1;
    auto ray = camera->RayAt(x, y, states + idx);
    auto temp = Trace(world, ray, states + idx);
    color += temp;
  }
  color /= float(spp);
  color = glm::clamp(color, 0.f, 1.f);
  // gamma correction
  color = glm::sqrt(color);
  out_image[idx] = color;
}