#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif

#include <glog/logging.h>
#include <stb_image_write.h>

#include <fstream>
#include <glm/glm.hpp>
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

std::string BaseName(const std::string &path) {
  int i;
  for (i = (int)path.length() - 1; i >= 0; i--)
    if (path[i] == '\\' || path[i] == '/') break;
  return path.substr(i + 1);
}

std::string ParentPath(const std::string &path) {
  int i;
  for (i = path.size() - 2; i >= 0; i--) {
    if (path[i] == '/' || path[i] == '\\' && path[i + 1] == '\\') {
      break;
    }
  }
  return path.substr(0, i);
}

__global__ void CudaRandomInit(uint64_t seed, curandState *state, int n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= n) return;
  curand_init(seed, idx, 0, &state[idx]);
}

__device__ bool TriangleHit(const vec3 p[3], const Ray &ray, double t_from,
                            double t_to, double *out_t, vec3 *out_normal,
                            double *out_u, double *out_v) {
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
  *out_u = u;
  *out_v = v;

  return true;
}

void WriteImage(const std::vector<vec3> &pixels, int height, int width,
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

__host__ __device__ int GetWorkload(int rank, int world_size, int spp) {
  return spp / world_size + (int)(rank < (spp % world_size));
}

__host__ void GatherImageData(int height, int width,
                              std::vector<glm::vec3> &send_buf, int spp,
                              std::vector<glm::vec3> *out_image) {
  int rank, world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  out_image->resize(height * width);
  for (int i = 0; i < height * width; i++) out_image->at(i) = glm::vec3(0);
  MPI_Reduce(send_buf.data(), out_image->data(), height * width * 3, MPI_FLOAT,
             MPI_SUM, 0, MPI_COMM_WORLD);
  for (int i = 0; i < height * width; i++) {
    out_image->at(i) =
        glm::sqrt(glm::clamp(out_image->at(i) / (float)spp, 0.f, 1.f));
  }
}

__host__ void Main(
    curandState **d_states, Camera **d_camera, HitableList **d_world,
    glm::vec3 **d_image,
    nvstd::function<void(HitableList *world, Camera *camera)> init_world,
    int height, int width, int spp) {
  cudaError err;

  cudaMalloc(d_states, sizeof(curandState) * height * width);
  cudaMalloc(d_image, sizeof(glm::vec3) * height * width);
  cudaMalloc(d_world, sizeof(HitableList));
  cudaMalloc(d_camera, sizeof(Camera));
  err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  CudaRandomInit<<<(height * width + 63) / 64, 64>>>(1024, *d_states,
                                                     height * width);
  init_world(*d_world, *d_camera);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 block(8, 8);
    dim3 grid((height + block.x - 1) / block.x,
              (width + block.y - 1) / block.y);
    cudaEventRecord(start);
    PathTracing<<<grid, block>>>(*d_world, *d_camera, height, width, spp, true,
                                 *d_states, *d_image);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    err = cudaGetLastError();
    CHECK(err == cudaSuccess) << cudaGetErrorString(err);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    LOG(INFO) << "Ray tracing finished in " << ms << "ms.";
  }

  std::vector<glm::vec3> image(height * width);
  err = cudaMemcpy(image.data(), *d_image, sizeof(glm::vec3) * height * width,
                   cudaMemcpyDeviceToHost);
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  WriteImage(image, height, width, "image.jpeg");
}

__host__ void DistributedMain(
    curandState **d_states, Camera **d_camera, HitableList **d_world,
    glm::vec3 **d_image,
    nvstd::function<void(HitableList *world, Camera *camera)> init_world,
    int height, int width, int spp) {
  int rank, world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int local_spp = GetWorkload(rank, world_size, spp);
  LOG(INFO) << "[" << rank << " / " << world_size
            << "] workload: " << local_spp;
  cudaError err;

  cudaMalloc(d_image, sizeof(glm::vec3) * height * width);
  cudaMalloc(d_world, sizeof(HitableList));
  cudaMalloc(d_camera, sizeof(Camera));
  cudaMalloc(d_states, sizeof(curandState) * height * width);
  err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);
  LOG(INFO) << "cuda memory allocated";

  CudaRandomInit<<<(height * width + 63) / 64, 64>>>(10086, *d_states,
                                                     height * width);
  LOG(INFO) << "cuda random initialized";
  init_world(*d_world, *d_camera);
  LOG(INFO) << "world initialized";

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 block(8, 8);
    dim3 grid((height + block.x - 1) / block.x,
              (width + block.y - 1) / block.y);
    cudaEventRecord(start);
    PathTracing<<<grid, block>>>(*d_world, *d_camera, height, width, local_spp,
                                 false, *d_states, *d_image);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    err = cudaGetLastError();
    CHECK(err == cudaSuccess) << cudaGetErrorString(err);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    LOG(INFO) << "[" << rank << " / " << world_size
              << "] Ray tracing finished in " << ms << "ms.";
  }

  std::vector<glm::vec3> image(height * width);
  err = cudaMemcpy(image.data(), *d_image, sizeof(glm::vec3) * height * width,
                   cudaMemcpyDeviceToHost);
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  std::vector<glm::vec3> final_image;
  GatherImageData(height, width, image, spp, &final_image);
  if (rank == 0) {
    WriteImage(final_image, height, width, "image.jpeg");
  }
}