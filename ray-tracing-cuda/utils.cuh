#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mpi.h>
#include <stdint.h>

#include <glm/glm.hpp>
#include <nvfunctional>
#include <string>
#include <tuple>
#include <vector>

#include "camera.cuh"
#include "hitable_list.cuh"
#include "ray.cuh"

__inline__ __device__ float CudaRandomFloat(float min, float max,
                                            curandState *state) {
  // (min, max]
  float t = curand_uniform(state);
  return t * (max - min) + min;
}

__global__ void CudaRandomInit(uint64_t seed, curandState *state, int n);

void WriteImage(const std::vector<glm::vec3> &pixels, int height, int width,
                const std::string &path);

__device__ bool TriangleHit(const glm::vec3 p[3], const Ray &ray, double t_from,
                            double t_to, double *out_t, glm::vec3 *out_normal,
                            double *out_u, double *out_v);

struct Layer;
__device__ void DebugTracePath(Layer *layers, int n);

__host__ __device__ void GetWorkload(int height, int width, int rank,
                                     int world_size, int *out_i_from,
                                     int *out_j_from, int *out_height_per_proc,
                                     int *out_width_per_proc);

__host__ void GatherImageData(int height, int width,
                              const std::vector<glm::vec3> &send_buf,
                              std::vector<glm::vec3> *out_image);

__host__ void Main(
    curandState **d_states, Camera **d_camera, HitableList **d_world,
    glm::vec3 **d_image,
    nvstd::function<void(HitableList *world, Camera *camera)> init_world,
    int height, int width, int spp);

__host__ void DistributedMain(
    curandState **d_states, Camera **d_camera, HitableList **d_world,
    glm::vec3 **d_image,
    nvstd::function<void(HitableList *world, Camera *camera)> init_world,
    int height, int width, int spp);