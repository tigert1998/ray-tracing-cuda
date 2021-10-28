#pragma once

#include <glm/glm.hpp>

#include "hitable.cuh"

struct Face {
  glm::vec3 points[3];
};

struct BV : public Hitable {
  glm::vec2 x, y, z;
  __device__ bool Hit(const Ray &ray, double t_from, double t_to,
                      HitRecord *out) override;
};

class BVHNode : public Hitable {
 private:
  Face *faces_;
  int n_, mid_;
  BVHNode *left_ = nullptr, *right_ = nullptr;
  BV bv_;

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

 public:
  __device__ explicit BVH(Face *faces, int n);
  __device__ bool Hit(const Ray &ray, double t_from, double t_to,
                      HitRecord *out) override;
};