#define STB_IMAGE_IMPLEMENTATION

#include <stb_image.h>

#include <cstdio>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "camera.cuh"
#include "diffuse_light.cuh"
#include "hitable_list.cuh"
#include "lambertian.cuh"
#include "parallelepiped.cuh"
#include "parallelogram.cuh"
#include "ray_tracing.cuh"
#include "sky.cuh"
#include "sphere.cuh"
#include "textures/constant_texture.cuh"
#include "textures/image_texture.cuh"
#include "utils.cuh"

const int WIDTH = 1280, HEIGHT = 720;

curandState *d_states;

Camera *d_camera;
HitableList *d_world;
glm::vec3 *d_image;
uint8_t *d_tex_image;
cudaTextureObject_t tex;

using glm::pi;
using glm::rotateX;
using glm::rotateY;
using glm::vec3;

__global__ void InitWorld(HitableList *world, Camera *camera,
                          cudaTextureObject_t tex) {
  new (world) HitableList();
  new (camera) Camera(vec3(278, 278, -800), vec3(278, 278, 0), vec3(0, 1, 0),
                      pi<double>() * 2 / 9, double(WIDTH) / HEIGHT);

  auto red_material_ptr = new Lambertian(vec3(0.65, 0.05, 0.05));
  auto white_material_ptr = new Lambertian(vec3(0.73, 0.73, 0.73));
  auto green_material_ptr = new Lambertian(vec3(0.12, 0.45, 0.15));

  auto light_material_ptr =
      new DiffuseLight(new ConstantTexture(vec3(1, 1, 1)));
  auto earthmap_texture_ptr = new ImageTexture(tex);
  auto earthmap_material_ptr = new Lambertian(earthmap_texture_ptr);
  auto sky = new Sky();

  world->Append(sky);
  vec3 parallelograms[] = {
      vec3(0, 0, 0),       vec3(0, 555, 0),     vec3(0, 0, 555),
      vec3(555, 0, 0),     vec3(555, 555, 0),   vec3(555, 0, 555),
      vec3(213, 554, 332), vec3(213, 554, 227), vec3(343, 554, 332),
      vec3(0, 0, 0),       vec3(555, 0, 0),     vec3(0, 0, 555),
      vec3(555, 555, 555), vec3(0, 555, 555),   vec3(555, 0, 555),
      vec3(0, 555, 0),     vec3(555, 555, 0),   vec3(0, 555, 555)};
  world->Append(new Parallelogram(&parallelograms[0], red_material_ptr));
  world->Append(new Parallelogram(&parallelograms[3], green_material_ptr));
  world->Append(new Parallelogram(&parallelograms[6], light_material_ptr));
  world->Append(new Parallelogram(&parallelograms[9], white_material_ptr));
  world->Append(new Parallelogram(&parallelograms[12], white_material_ptr));
  world->Append(new Parallelogram(&parallelograms[15], white_material_ptr));

  world->Append(new Sphere(vec3(278, 278, 0), 100, earthmap_material_ptr));
}

cudaTextureObject_t InitImageTextures() {
  int width, height, channels;
  uint64_t pitch_in_bytes;
  auto data =
      stbi_load("resources/earthmap.jpg", &width, &height, &channels, 4);
  cudaMallocPitch(&d_tex_image, &pitch_in_bytes, 4 * width, height);
  cudaMemcpy2D(d_tex_image, pitch_in_bytes, data, 4 * width, 4 * width, height,
               cudaMemcpyHostToDevice);
  auto tex = ImageTexture::CreateCudaTextureObj(d_tex_image, height, width,
                                                pitch_in_bytes);
  auto err = cudaGetLastError();
  CHECK(err == cudaSuccess) << cudaGetErrorString(err);
  return tex;
}

int main() {
  Main(
      &d_states, &d_camera, &d_world, &d_image,
      [](HitableList *world, Camera *camera) {
        auto tex = InitImageTextures();
        InitWorld<<<1, 1>>>(world, camera, tex);
      },
      HEIGHT, WIDTH, 200);
  return 0;
}