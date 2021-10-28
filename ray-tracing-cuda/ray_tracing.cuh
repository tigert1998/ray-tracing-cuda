#pragma once

#include <glm/glm.hpp>

#include "camera.cuh"
#include "hitable_list.cuh"
#include "ray.cuh"

struct Layer {
  glm::vec3 pos;
  double t;
  glm::vec3 target;
  glm::vec3 emitted;
  glm::vec3 attenuation;
};

__device__ glm::vec3 Trace(HitableList *world, Ray ray);

__global__ void RayTracing(HitableList *world, Camera *camera, int height,
                           int width, int spp, curandState *states,
                           glm::vec3 *out_image);

__global__ void DistributedRayTracing(int rank, int world_size,
                                      HitableList *world, Camera *camera,
                                      int height, int width, int spp,
                                      curandState *states,
                                      glm::vec3 *out_image);