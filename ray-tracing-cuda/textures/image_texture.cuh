#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>

#include "textures/texture.cuh"

class ImageTexture : public Texture {
 private:
  cudaTextureObject_t image_texture_;

 public:
  __host__ __device__ ImageTexture(cudaTextureObject_t image_texture);
  __device__ glm::vec3 Value(double u, double v, const glm::vec3 &p) const;

  static cudaTextureObject_t CreateCudaTextureObj(uint8_t *dev_buffer,
                                                  int height, int width,
                                                  uint64_t pitch_in_bytes);
};