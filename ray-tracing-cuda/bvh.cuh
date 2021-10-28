#pragma once

#include <glm/glm.hpp>

#include "hitable.cuh"

struct Face {
  glm::vec3 points[3];
};

struct BB : public Hitable {
 public:
  glm::vec3 min, max;
  __device__ bool Hit(const Ray &ray, double t_from, double t_to,
                      HitRecord *out) override;

 private:
  __device__ bool CheckOnPlane(const Ray &ray, double t, double t_from,
                               double t_to, int axis);
};

class BVHNode : public Hitable {
 private:
  // tuned for 2080 ti
  constexpr static int kMinFaces = 4096;

  Face *faces_;
  int n_, mid_;
  BVHNode *left_ = nullptr, *right_ = nullptr;
  BB bb_;

 public:
  __device__ explicit BVHNode(Face *faces, int n);
  __device__ bool Hit(const Ray &ray, double t_from, double t_to,
                      HitRecord *out) override;
};

class BVH : public Hitable {
 private:
  Face *faces_;
  int n_;
  BVHNode *root_;
  Material *material_ptr_;

 public:
  __device__ explicit BVH(Face *faces, int n, Material *material_ptr);
  __device__ bool Hit(const Ray &ray, double t_from, double t_to,
                      HitRecord *out) override;
};