#include <cstdio>
#include <iostream>

#include "camera.cuh"
#include "hitable_list.cuh"
#include "lambertian.cuh"
#include "sphere.cuh"
#include "textures/constant_texture.cuh"
#include "utils.cuh"

const int WIDTH = 1280, HEIGHT = 720;

curandState *d_states;
ConstantTexture *d_const_tex;
Lambertian *d_lambertian;
Sphere *d_sphere_0, *d_sphere_1;
Camera *d_camera;
HitableList *d_world;

int main() {
  cudaMalloc(&d_const_tex, sizeof(ConstantTexture));
  cudaMalloc(&d_states, sizeof(curandState) * 2);
  cudaMalloc(&d_lambertian, sizeof(Lambertian));
  cudaMalloc(&d_sphere_0, sizeof(Sphere));
  cudaMalloc(&d_sphere_1, sizeof(Sphere));
  cudaMalloc(&d_camera, sizeof(Camera));
  cudaMalloc(&d_world, sizeof(HitableList));

  ConstantTexture const_tex(glm::vec3(0, 1, 0));
  const_tex.ToDevice(d_const_tex);
  Lambertian lambertian(&d_states[0], d_const_tex);
  lambertian.ToDevice(d_lambertian);
  Sphere sphere_0(glm::vec3(0, 0, -1), 0.5, d_lambertian);
  sphere_0.ToDevice(d_sphere_0);
  Sphere sphere_1(glm::vec3(0, -100.5, -1), 100, d_lambertian);
  sphere_1.ToDevice(d_sphere_1);
  Camera camera(glm::vec3(3, 3, 2), glm::vec3(0, 0, -1), glm::vec3(0, 1, 0),
                glm::radians(20), WIDTH * 1.0 / HEIGHT, &d_states[1]);
  camera.ToDevice(d_camera);
  HitableList world;
  world.Append(d_sphere_0);
  world.Append(d_sphere_1);
  world.ToDevice(d_world);

  CudaRandomInit<<<2, 1>>>(10086, d_states);

  return 0;
}