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
ConstantTexture *d_const_tex;
Lambertian *d_lambertian;
Sphere *d_sphere_0, *d_sphere_1;
Sky *d_sky;
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

int main() {
  cudaError err;

  cudaMalloc(&d_const_tex, sizeof(ConstantTexture));
  cudaMalloc(&d_states, sizeof(curandState) * WIDTH * HEIGHT);
  cudaMalloc(&d_lambertian, sizeof(Lambertian));
  cudaMalloc(&d_sphere_0, sizeof(Sphere));
  cudaMalloc(&d_sphere_1, sizeof(Sphere));
  cudaMalloc(&d_sky, sizeof(Sky));
  cudaMalloc(&d_camera, sizeof(Camera));
  cudaMalloc(&d_world, sizeof(HitableList));
  cudaMalloc(&d_image, sizeof(glm::vec3) * WIDTH * HEIGHT);
  err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);

  ConstantTexture const_tex(glm::vec3(0, 1, 0));
  const_tex.ToDevice(d_const_tex);
  Lambertian lambertian(&d_states[0], d_const_tex);
  lambertian.ToDevice(d_lambertian);
  Sphere sphere_0(glm::vec3(0, 0, -1), 0.5, d_lambertian);
  sphere_0.ToDevice(d_sphere_0);
  Sphere sphere_1(glm::vec3(0, -100.5, -1), 100, d_lambertian);
  sphere_1.ToDevice(d_sphere_1);
  Sky sky;
  sky.ToDevice(d_sky);
  Camera camera(glm::vec3(3, 3, 2), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0),
                glm::radians(20), WIDTH * 1.0 / HEIGHT, &d_states[1]);
  camera.ToDevice(d_camera);
  HitableList world;
  world.Append(d_sphere_0);
  world.Append(d_sphere_1);
  world.Append(d_sky);
  world.ToDevice(d_world);

  CudaRandomInit<<<WIDTH * HEIGHT / 64, 64>>>(10086, d_states);
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