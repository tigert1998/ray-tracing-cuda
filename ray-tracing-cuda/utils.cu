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

__global__ void CudaRandomInit(uint64_t seed, curandState *state, int n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= n) return;
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

__host__ __device__ void GetWorkload(int height, int width, int rank,
                                     int world_size, int *out_i_from,
                                     int *out_j_from, int *out_height_per_proc,
                                     int *out_width_per_proc) {
  int div = width / world_size;
  int mod = width % world_size;
  *out_height_per_proc = height;
  *out_width_per_proc = div + (rank < mod);
  *out_i_from = 0;
  if (rank < mod) {
    *out_j_from = (div + 1) * rank;
  } else {
    *out_j_from = (div + 1) * mod + div * (rank - mod);
  }
}

__host__ void GatherImageData(int height, int width,
                              const std::vector<glm::vec3> &send_buf,
                              std::vector<glm::vec3> *out_image) {
  int rank, world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> i_froms(world_size), j_froms(world_size),
      height_per_procs(world_size), width_per_procs(world_size),
      dis(world_size), counts(world_size);

  for (int i = 0; i < world_size; i++) {
    GetWorkload(height, width, i, world_size, &i_froms[i], &j_froms[i],
                &height_per_procs[i], &width_per_procs[i]);
    counts[i] = height_per_procs[i] * width_per_procs[i] * sizeof(glm::vec3);
    if (i == 0) {
      dis[0] = 0;
    } else {
      dis[i] = dis[i - 1] + counts[i - 1];
    }
  }

  std::vector<glm::vec3> gathered(height * width);
  MPI_Gatherv(send_buf.data(), counts[rank], MPI_BYTE, gathered.data(),
              counts.data(), dis.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    out_image->resize(height * width);
    for (int other = 0; other < world_size; other++) {
      for (int i = 0; i < height_per_procs[other]; i++)
        for (int j = 0; j < width_per_procs[other]; j++) {
          int idx =
              i * width_per_procs[other] + j + dis[other] / sizeof(glm::vec3);
          int to_idx = (i_froms[other] + i) * width + (j_froms[other] + j);
          out_image->operator[](to_idx) = gathered[idx];
        }
    }
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
    RayTracing<<<grid, block>>>(*d_world, *d_camera, height, width, spp,
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
  int rank, world_size, i_from, j_from, height_per_proc, width_per_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  GetWorkload(height, width, rank, world_size, &i_from, &j_from,
              &height_per_proc, &width_per_proc);
  LOG(INFO) << "[" << rank << " / " << world_size << "] workload: (" << i_from
            << ", " << j_from << ", " << i_from + height_per_proc << ", "
            << j_from + width_per_proc << ")";

  cudaError err;

  cudaMalloc(d_states, sizeof(curandState) * width_per_proc * height_per_proc);
  cudaMalloc(d_image, sizeof(glm::vec3) * width_per_proc * height_per_proc);
  cudaMalloc(d_world, sizeof(HitableList));
  cudaMalloc(d_camera, sizeof(Camera));
  err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  CudaRandomInit<<<(width_per_proc * height_per_proc + 63) / 64, 64>>>(
      10086, *d_states, height_per_proc * width_per_proc);
  init_world(*d_world, *d_camera);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 block(8, 8);
    dim3 grid((height_per_proc + block.x - 1) / block.x,
              (width_per_proc + block.y - 1) / block.y);
    cudaEventRecord(start);
    DistributedRayTracing<<<grid, block>>>(rank, world_size, *d_world,
                                           *d_camera, height, width, spp,
                                           *d_states, *d_image);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    err = cudaGetLastError();
    CHECK(err == cudaSuccess) << cudaGetErrorString(err);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    LOG(INFO) << "[" << rank << " / " << world_size
              << "] Ray tracing finished in " << ms << "ms.";
  }

  std::vector<glm::vec3> image(height_per_proc * width_per_proc);
  err = cudaMemcpy(image.data(), *d_image,
                   sizeof(glm::vec3) * height_per_proc * width_per_proc,
                   cudaMemcpyDeviceToHost);
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  std::vector<glm::vec3> final_image;
  GatherImageData(height, width, image, &final_image);
  if (rank == 0) {
    WriteImage(final_image, height, width, "image.jpeg");
  }
}