#include "image_texture.cuh"

using glm::clamp;

__host__ __device__
ImageTexture::ImageTexture(cudaTextureObject_t image_texture)
    : image_texture_(image_texture) {}

__device__ glm::vec3 ImageTexture::Value(double u, double v,
                                         const glm::vec3 &p) const {
  v = 1.0 - v;  // Flip V to image coordinates

  float4 ret = tex2D<float4>(image_texture_, u, v);
  return glm::vec3(ret.x, ret.y, ret.z);
}

cudaTextureObject_t ImageTexture::CreateCudaTextureObj(
    uint8_t *dev_buffer, int height, int width, uint64_t pitch_in_bytes) {
  cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypePitch2D;
  res_desc.res.pitch2D.devPtr = dev_buffer;
  res_desc.res.pitch2D.desc.f = cudaChannelFormatKindUnsigned;
  res_desc.res.pitch2D.desc.x = 8;
  res_desc.res.pitch2D.desc.y = 8;
  res_desc.res.pitch2D.desc.z = 8;
  res_desc.res.pitch2D.desc.w = 8;
  res_desc.res.pitch2D.height = height;
  res_desc.res.pitch2D.width = width;
  res_desc.res.pitch2D.pitchInBytes = pitch_in_bytes;
  cudaTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.readMode = cudaReadModeNormalizedFloat;
  tex_desc.normalizedCoords = 1;
  cudaTextureObject_t tex;
  cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr);
  return tex;
}
