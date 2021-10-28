#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif

#include <glog/logging.h>
#include <stb_image_write.h>

#include <fstream>
#include <glm/gtx/intersect.hpp>
#include <string>

#include "hitable.cuh"
#include "ray.cuh"
#include "ray_tracing.cuh"
#include "utils.cuh"

using glm::cross;
using glm::dot;
using glm::normalize;
using glm::vec3;

__global__ void CudaRandomInit(uint64_t seed, curandState *state) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}

__device__ bool TriangleHit(const vec3 p[3], const Ray &ray, double t_from,
                            double t_to, double *out_t, vec3 *out_normal) {
  //  MOLLER_TRUMBORE
  double eps = 1e-7;
  vec3 v0v1 = p[1] - p[0];
  vec3 v0v2 = p[2] - p[0];
  vec3 pvec = cross(ray.direction(), v0v2);
  float det = dot(v0v1, pvec);

  // ray and triangle are parallel if det is close to 0
  if (fabs(det) < eps) return false;

  float invDet = 1 / det;

  vec3 tvec = ray.position() - p[0];
  float u = dot(tvec, pvec) * invDet;
  if (u < 0 || u > 1) return false;

  vec3 qvec = cross(tvec, v0v1);
  float v = dot(ray.direction(), qvec) * invDet;
  if (v < 0 || u + v > 1) return false;

  float t = dot(v0v2, qvec) * invDet;

  if (!(t_from <= t && t <= t_to)) {
    return false;
  }

  *out_t = t;
  vec3 n = normalize(cross(v0v1, v0v2));
  *out_normal = dot(ray.direction(), n) < 0 ? n : -n;

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

__device__ void DebugTracePath(Layer *layers, int n) {
  for (int i = 0; i < n; i++) {
    printf(
        "[%d] (%.3f, %.3f, %.3f) -> (%.3f, %.3f, %.3f) with t: %.3f, "
        "attenuation: (%.3f, %.3f, %.3f), emitted: (%.3f, %.3f, %.3f)\n",
        i, layers[i].pos[0], layers[i].pos[1], layers[i].pos[2],
        layers[i].target[0], layers[i].target[1], layers[i].target[2],
        layers[i].t, layers[i].attenuation[0], layers[i].attenuation[1],
        layers[i].attenuation[2], layers[i].emitted[0], layers[i].emitted[1],
        layers[i].emitted[2]);
  }
}