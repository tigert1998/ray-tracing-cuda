#pragma once

#include <glm/glm.hpp>
#include <nvfunctional>

#include "hitable_list.cuh"

//    1
//   /|\
//  ? 0 ?
//  |X X|
//  2 ? 3
//   \|/
//    ?

class Parallelepiped : public Hitable {
 private:
  HitableList list_;
  __device__ __host__ void AddCorner(glm::vec3 p[4], Material* material_ptr);

 public:
  Parallelepiped() = delete;
  __host__ __device__ Parallelepiped(glm::vec3 p[4], Material* material_ptr);
  __host__ __device__
  Parallelepiped(glm::vec3 lengths, Material* material_ptr,
                 nvstd::function<glm::vec3(glm::vec3)> transform);
  __device__ bool Hit(const Ray& ray, double t_from, double t_to,
                      HitRecord* out) override;
};