#include <curand_kernel.h>

#include <tuple>

#include "ray_tracing.cuh"
#include "utils.cuh"

using glm::vec3;

constexpr int TRACE_DEPTH_LIMIT = 10;

__device__ glm::vec3 Trace(HitableList *world, Ray ray, curandState *states) {
  Layer storage[TRACE_DEPTH_LIMIT];

  int storage_size = 0;

  bool should_debug = false;

  vec3 result;
  for (int depth = 0;; depth++) {
    HitRecord record;
    bool hit = world->Hit(ray, 1e-3, INFINITY, &record);
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
    auto emitted = material_ptr->Emit(record.u, record.v, hit_point);
    if (!scattered) {
      result = emitted;
      break;
    }
    storage[storage_size].t = record.t;
    storage[storage_size].pos = ray.position();
    storage[storage_size].target = hit_point;
    storage[storage_size].emitted = emitted;
    storage[storage_size++].attenuation = attenuation;
    ray = reflection;
  }

  if (should_debug) {
    DebugTracePath(storage, storage_size);
  }

  for (int i = storage_size - 1; i >= 0; i--) {
    result = storage[i].emitted + storage[i].attenuation * result;
  }
  return result;
}

__global__ void PathTracing(HitableList *world, Camera *camera, int height,
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

__global__ void DistributedPathTracing(int rank, int world_size,
                                       HitableList *world, Camera *camera,
                                       int height, int width, int spp,
                                       curandState *states, vec3 *out_image) {
  // 0 <= i < height_per_proc
  // 0 <= j < width_per_proc
  int i_from, j_from, h_per_proc, w_per_proc;
  GetWorkload(height, width, rank, world_size, &i_from, &j_from, &h_per_proc,
              &w_per_proc);

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= h_per_proc || j >= w_per_proc) return;
  int idx = i * w_per_proc + j;

  auto color = vec3(0);
  for (int k = 0; k < spp; k++) {
    double x = (CudaRandomFloat(0, 1, states + idx) + double(j + j_from)) /
               double(width);
    double y =
        (CudaRandomFloat(0, 1, states + idx) + double(height - (i + i_from))) /
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