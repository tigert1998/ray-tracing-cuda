#include "ray.cuh"

using glm::normalize;
using glm::vec3;

__device__ __host__ Ray::Ray() = default;

__device__ __host__ Ray::Ray(vec3 position, vec3 direction)
    : position_(position) {
  direction_ = normalize(direction);
}

__device__ __host__ vec3 Ray::position() const { return position_; }

__device__ __host__ vec3 Ray::direction() const { return direction_; }
