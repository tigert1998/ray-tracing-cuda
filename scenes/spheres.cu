#include <cstdio>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <iostream>
#include <string>
#include <vector>

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

Camera *d_camera;
HitableList *d_world;
glm::vec3 *d_image;
curandState *d_states;

using glm::pi;
using glm::vec3;

__global__ void InitWorld(HitableList *world, Camera *camera,
                          curandState *state) {
  new (world) HitableList();
  new (camera) Camera(vec3(13, 2, 3), vec3(0, 0, 0), vec3(0, 1, 0),
                      pi<double>() * 1 / 9, double(WIDTH) / HEIGHT);

  auto ground_material = new Lambertian(vec3(0.5, 0.5, 0.5));
  world->Append(new Sphere(vec3(0, -1000, 0), 1000, ground_material));

  auto material1 = new Dielectric(vec3(1, 1, 1), 1.5);
  world->Append(new Sphere(vec3(0, 1, 0), 1.0, material1));

  auto material2 = new Lambertian(vec3(0.4, 0.2, 0.1));
  world->Append(new Sphere(vec3(-4, 1, 0), 1.0, material2));

  auto material3 = new Metal(vec3(0.7, 0.6, 0.5));
  world->Append(new Sphere(vec3(4, 1, 0), 1.0, material3));

  world->Append(new Sky());

  for (int a = -3; a < 3; a++) {
    for (int b = -3; b < 3; b++) {
      auto choose_mat = CudaRandomFloat(0, 1, state);
      vec3 center(a + CudaRandomFloat(0, 0.9, state), 0.2,
                  b + CudaRandomFloat(0, 0.9, state));

      if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
        Material *sphere_material;

        auto albedo =
            vec3(CudaRandomFloat(0, 1, state), CudaRandomFloat(0, 1, state),
                 CudaRandomFloat(0, 1, state));
        if (choose_mat < 0.8) {
          // diffuse
          sphere_material = new Lambertian(albedo);
        } else if (choose_mat < 0.95) {
          // metal
          auto fuzz = CudaRandomFloat(0, 0.5, state);
          sphere_material = new Metal(albedo, fuzz);
        } else {
          // glass
          sphere_material = new Dielectric(vec3(1, 1, 1), 1.5);
        }
        world->Append(new Sphere(center, 0.2, sphere_material));
      }
    }
  }
}

int main() {
  Main(
      &d_states, &d_camera, &d_world, &d_image,
      [](HitableList *world, Camera *camera) {
        InitWorld<<<1, 1>>>(world, camera, d_states);
      },
      HEIGHT, WIDTH, 200);
  return 0;
}