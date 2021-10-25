#pragma once

#include <glm/glm.hpp>

#include "textures/texture.cuh"

class ConstantTexture : public Texture {
 private:
  glm::vec3 color_;

 public:
  ConstantTexture() = delete;
  ConstantTexture(glm::vec3 color);
  __device__ glm::vec3 Value(double u, double v, const glm::vec3 &p) const;
};