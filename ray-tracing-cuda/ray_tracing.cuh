#pragma once

#include <glm/glm.hpp>

#include "camera.cuh"
#include "hitable_list.cuh"
#include "ray.cuh"

__device__ glm::vec3 Trace(HitableList *world, Ray ray);

__global__ void RayTracing(HitableList *world, Camera *camera) {}