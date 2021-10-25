#pragma once

#include <glm/glm.hpp>

class Ray {
 public:
  Ray() = delete;
  __device__ __host__ Ray(glm::vec3 position, glm::vec3 direction);
  __device__ __host__ glm::vec3 position() const;
  __device__ __host__ glm::vec3 direction() const;

 private:
  glm::vec3 position_, direction_;
};
