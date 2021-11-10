#include <assimp/cimport.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <mpi.h>

#include <cstdio>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <iostream>
#include <string>
#include <tuple>
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
using std::tuple;

__global__ void InitWorld(HitableList *world, Camera *camera) {
  new (world) HitableList();
  new (camera)
      Camera(vec3(-0.025, 0.1, -0.5), vec3(-0.025, 0.1, 0), vec3(0, 1, 0),
             pi<double>() * 2 / 9, double(WIDTH) / HEIGHT);
  auto sky = new Sky();
  vec3 parallelograms[] = {vec3(-0.025 - 0.5, 0.1 - 0.5, 2),
                           vec3(-0.025 + 0.5, 0.1 - 0.5, 2),
                           vec3(-0.025 - 0.5, 0.1 + 0.5, 2)};
  auto green_material_ptr = new Lambertian(vec3(0.12, 0.45, 0.15));
  world->Append(new Parallelogram(&parallelograms[0], green_material_ptr));
  world->Append(sky);
}

__global__ void InitModel(HitableList *world, Face *faces, int n) {
  auto white_material_ptr = new Metal(vec3(1, 1, 1), 0.5);
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
  auto err = cudaMalloc(&d_faces, sizeof(Face) * faces.size());
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);
  err = cudaMemcpy(d_faces, faces.data(), sizeof(Face) * faces.size(),
                   cudaMemcpyHostToDevice);
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);
  InitModel<<<1, 1>>>(d_world, d_faces, faces.size());
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);
}

void InitCudaAndMPI(int argc, char **argv) {
  int rank, world_size;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int device_count;
  cudaGetDeviceCount(&device_count);
  int device = rank % device_count;
  cudaSetDevice(device);
  LOG(INFO) << "[" << rank << " / " << world_size
            << "] binding to cuda:" << device;
  auto err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);
}

int main(int argc, char **argv) {
  InitCudaAndMPI(argc, argv);
  DistributedMain(
      &d_states, &d_camera, &d_world, &d_image,
      [](HitableList *world, Camera *camera) {
        InitWorld<<<1, 1>>>(world, camera);
        ImportModel("bunny.obj");
      },
      HEIGHT, WIDTH, 20);
  MPI_Finalize();
  return 0;
}