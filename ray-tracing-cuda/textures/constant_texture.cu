#include "textures/constant_texture.cuh"

using glm::vec3;

__host__ __device__ ConstantTexture::ConstantTexture() = default;

__host__ __device__ ConstantTexture::ConstantTexture(glm::vec3 color) {
  color_ = color;
}

__device__ glm::vec3 ConstantTexture::Value(double u, double v,
                                            const glm::vec3 &p) const {
  return color_;
}