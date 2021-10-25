#include "textures/constant_texture.cuh"

using glm::vec3;

ConstantTexture::ConstantTexture(glm::vec3 color) { color_ = color; }

__device__ glm::vec3 ConstantTexture::Value(double u, double v,
                                            const glm::vec3 &p) const {
  return color_;
}