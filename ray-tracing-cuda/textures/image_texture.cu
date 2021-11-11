#include "image_texture.cuh"

using glm::clamp;

__host__ __device__ ImageTexture::ImageTexture(int height, int width,
                                               int components, uint8_t *data)
    : height_(height), width_(width), components_(components), data_(data) {}

__device__ glm::vec3 ImageTexture::Value(double u, double v,
                                         const glm::vec3 &p) const {
  u = clamp(u, 0.0, 1.0);
  v = 1.0 - clamp(v, 0.0, 1.0);  // Flip V to image coordinates

  auto i = static_cast<int>(u * width_);
  auto j = static_cast<int>(v * height_);

  // Clamp integer mapping, since actual coordinates should be less than 1.0
  if (i >= width_) i = width_ - 1;
  if (j >= height_) j = height_ - 1;

  uint8_t *pixel = data_ + j * (components_ * width_) + i * components_;

  return glm::vec3(pixel[0], pixel[1], pixel[2]) / 255.0f;
}
