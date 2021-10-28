#include <assimp/cimport.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <cstdio>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "bvh.cuh"
#include "camera.cuh"
#include "dielectric.cuh"
#include "diffuse_light.cuh"
#include "hitable_list.cuh"
#include "lambertian.cuh"
#include "metal.cuh"
#include "parallelepiped.cuh"
#include "parallelogram.cuh"
#include "ray_tracing.cuh"
#include "sky.cuh"
#include "sphere.cuh"
#include "textures/constant_texture.cuh"
#include "utils.cuh"

const int WIDTH = 1280, HEIGHT = 720;

curandState *d_states;

Camera *d_camera;
HitableList *d_world;
glm::vec3 *d_image;
Face *d_faces;

using glm::pi;
using glm::vec3;

__global__ void InitWorld(HitableList *world, Camera *camera) {
  new (world) HitableList();
  new (camera)
      Camera(vec3(-0.025, 0.1, -0.5), vec3(-0.025, 0.1, 0), vec3(0, 1, 0),
             pi<double>() * 2 / 9, double(WIDTH) / HEIGHT);
  auto sky = new Sky();
  world->Append(sky);
}

__global__ void InitModel(HitableList *world, Face *faces, int n) {
  auto white_material_ptr = new Metal(vec3(0.72, 0.72, 0.72));
  auto green_material_ptr = new Lambertian(vec3(0.12, 0.45, 0.15));
  vec3 parallelograms[] = {vec3(-0.025 - 0.5, 0.1 - 0.5, 1.2),
                           vec3(-0.025 + 0.5, 0.1 - 0.5, 1.2),
                           vec3(-0.025 - 0.5, 0.1 + 0.5, 1.2)};
  world->Append(new Parallelogram(&parallelograms[0], green_material_ptr));
  auto bvh = new BVH(faces, n, white_material_ptr);
  world->Append(bvh);
}

void ImportModel(const std::string &path) {
  const aiScene *scene = aiImportFile(
      path.c_str(), aiProcess_GlobalScale | aiProcess_CalcTangentSpace |
                        aiProcess_Triangulate);
  std::vector<Face> faces;
  for (int i = 0; i < scene->mNumMeshes; i++) {
    auto mesh = scene->mMeshes[i];
    faces.reserve(faces.capacity() + mesh->mNumFaces);
    for (int j = 0; j < mesh->mNumFaces; j++) {
      Face face;
      for (int k = 0; k < 3; k++) {
        int idx = mesh->mFaces[j].mIndices[k];
        auto vertex = mesh->mVertices[idx];
        face.points[k] = vec3(vertex.x, vertex.y, vertex.z);
      }
      faces.emplace_back(face);
    }
  }
  aiReleaseImport(scene);
  cudaMalloc(&d_faces, sizeof(Face) * faces.size());
  cudaMemcpy(d_faces, faces.data(), sizeof(Face) * faces.size(),
             cudaMemcpyHostToDevice);
  InitModel<<<1, 1>>>(d_world, d_faces, faces.size());
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);
}

int main() {
  cudaError err;

  cudaMalloc(&d_states, sizeof(curandState) * WIDTH * HEIGHT);
  cudaMalloc(&d_image, sizeof(glm::vec3) * WIDTH * HEIGHT);
  cudaMalloc(&d_world, sizeof(HitableList));
  cudaMalloc(&d_camera, sizeof(Camera));
  err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  InitWorld<<<1, 1>>>(d_world, d_camera);
  CudaRandomInit<<<WIDTH * HEIGHT / 64, 64>>>(10086, d_states);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  ImportModel("bunny.obj");

  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 block(8, 8);
    dim3 grid((HEIGHT + block.x - 1) / block.x,
              (WIDTH + block.y - 1) / block.y);
    cudaEventRecord(start);
    RayTracing<<<grid, block>>>(d_world, d_camera, HEIGHT, WIDTH, 20, d_states,
                                d_image);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    err = cudaGetLastError();
    CHECK(err == cudaSuccess) << cudaGetErrorString(err);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    LOG(INFO) << "Ray tracing finished in " << ms << "ms.";
  }

  std::vector<glm::vec3> image(HEIGHT * WIDTH);
  err = cudaMemcpy(image.data(), d_image, sizeof(glm::vec3) * HEIGHT * WIDTH,
                   cudaMemcpyDeviceToHost);
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  WriteImage(image, HEIGHT, WIDTH, "image.jpeg");
  return 0;
}