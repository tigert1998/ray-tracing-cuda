#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "camera.cuh"
#include "hitable_list.cuh"
#include "lambertian.cuh"
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

void Output(std::vector<glm::vec3> &pixels, int height, int width) {
  std::string title = "image.ppm";
  LOG(INFO) << "Writing to " << title << "..." << std::endl;
  std::fstream fs(title, std::ios::out);
  fs << "P3\n" << width << " " << height << "\n255\n";
  for (int i = 0; i < width * height; i++) {
    auto color = pixels[i];
    for (int j = 0; j < 3; j++) fs << int(color[j] * 255) << " ";
  }
}

__global__ void InitWorld(HitableList *world, Camera *camera,
                          curandState *states) {
  auto const_tex = new ConstantTexture(glm::vec3(0, 1, 0));
  auto lambertian = new Lambertian(&states[0], const_tex);
  auto sphere_0 = new Sphere(glm::vec3(0, 0, -1), 0.5, lambertian);
  auto sphere_1 = new Sphere(glm::vec3(0, -100.5, -1), 100, lambertian);
  auto sky = new Sky();

  new (camera)
      Camera(glm::vec3(3, 3, 2), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0),
             glm::radians(20), WIDTH * 1.0 / HEIGHT, &states[1]);

  new (world) HitableList();
  world->Append(sphere_0);
  world->Append(sphere_1);
  world->Append(sky);
}

int main() {
  cudaError err;

  cudaMalloc(&d_states, sizeof(curandState) * WIDTH * HEIGHT);
  cudaMalloc(&d_image, sizeof(glm::vec3) * WIDTH * HEIGHT);
  cudaMalloc(&d_world, sizeof(HitableList));
  cudaMalloc(&d_camera, sizeof(Camera));
  err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  CudaRandomInit<<<WIDTH * HEIGHT / 64, 64>>>(10086, d_states);
  InitWorld<<<1, 1>>>(d_world, d_camera, d_states);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  dim3 grid(HEIGHT / 8, WIDTH / 8);
  dim3 block(8, 8);
  RayTracing<<<grid, block>>>(d_world, d_camera, HEIGHT, WIDTH, 100, d_states,
                              d_image);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  std::vector<glm::vec3> image(HEIGHT * WIDTH);
  err = cudaMemcpy(image.data(), d_image, sizeof(glm::vec3) * HEIGHT * WIDTH,
                   cudaMemcpyDeviceToHost);
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  Output(image, HEIGHT, WIDTH);
  return 0;
}