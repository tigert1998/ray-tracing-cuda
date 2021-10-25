#pragma once

#include <curand_kernel.h>

#include <glm/glm.hpp>

#include "cuda_copyable.cuh"
#include "ray.cuh"

class Camera : public CudaCopyable {
 private:
  glm::vec3 position_, lower_left_corner_, horizontal_, vertical_, u_, v_, w_;
  bool is_defocus_camera_;
  double lens_radius_ = -1;
  curandState *state_;

  __device__ glm::vec2 DiskRand(float radius);

 public:
  Camera() = delete;
  Camera(glm::vec3 position, glm::vec3 look_at, glm::vec3 up,
         double field_of_view, double width_height_aspect, curandState *state);
  Camera(glm::vec3 position, glm::vec3 look_at, glm::vec3 up,
         double field_of_view, double width_height_aspect, double aperture,
         double focus_distance, curandState *state);
  Camera(glm::vec3 position, glm::vec3 lower_left_corner, glm::vec3 horizontal,
         glm::vec3 vertical, curandState *state);
  __device__ glm::vec3 position() const;
  __device__ glm::vec3 lower_left_corner() const;
  __device__ glm::vec3 horizontal() const;
  __device__ glm::vec3 vertical() const;
  __device__ Ray RayAt(double x, double y);
  __device__ bool is_defocus_camera() const;
};
