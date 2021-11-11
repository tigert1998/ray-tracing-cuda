#pragma once

#include <glm/glm.hpp>

#include "textures/texture.cuh"

class ImageTexture : public Texture {
 private:
  uint8_t *data_;
  int height_, width_, components_;

 public:
  __host__ __device__ ImageTexture(int height, int width, int components,
                                   uint8_t *data);
  __device__ glm::vec3 Value(double u, double v, const glm::vec3 &p) const;
};