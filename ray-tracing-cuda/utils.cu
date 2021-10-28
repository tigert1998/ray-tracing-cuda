#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif

#include <glog/logging.h>
#include <stb_image_write.h>

#include <fstream>
#include <string>

#include "hitable.cuh"
#include "ray.cuh"
#include "utils.cuh"

using glm::cross;
using glm::dot;
using glm::normalize;
using glm::vec3;

__global__ void CudaRandomInit(uint64_t seed, curandState *state) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}

__device__ bool TriangleHit(vec3 p[3], const Ray &ray, double t_from,
                            double t_to, double *out_t, vec3 *out_normal) {
  vec3 n = normalize(cross(p[2] - p[0], p[1] - p[2]));
  double den = dot(ray.direction(), n);
  double num = dot(p[0] - ray.position(), n);
  double t = num / den;
  if (std::isnan(t) || std::isinf(t)) return false;
  if (t < t_from || t > t_to) return false;
  vec3 hit_point = ray.position() + (float)t * ray.direction();
  {
    double dot_values[3];
    for (int i = 0; i < 3; i++) {
      vec3 a = p[i], b = p[(i + 1) % 3];
      dot_values[i] = dot(cross(a - hit_point, b - hit_point), n);
    }
    for (int i = 0; i < 3; i++)
      if (dot_values[i] * dot_values[(i + 1) % 3] < 0) return false;
  }
  *out_t = t;
  *out_normal = dot(n, ray.direction()) < 0 ? n : -n;
  return true;
}

void WriteImage(std::vector<vec3> &pixels, int height, int width,
                const std::string &path) {
  std::vector<uint8_t> data;
  data.resize(pixels.size() * 3);
  for (int i = 0; i < pixels.size(); i++) {
    for (int j = 0; j < 3; j++) data[i * 3 + j] = pixels[i][j] * 255;
  }
  LOG(INFO) << "Writing image to " << path << "...";
  stbi_write_jpg(path.c_str(), width, height, 3, data.data(), 100);
}
