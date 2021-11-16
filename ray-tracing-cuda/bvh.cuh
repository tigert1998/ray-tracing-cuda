#pragma once

#include <thrust/sort.h>

#include <glm/glm.hpp>

#include "hitable.cuh"
#include "utils.cuh"

struct AABB : public Hitable {
 public:
  glm::vec3 min, max;
  __device__ bool Hit(const Ray &ray, double t_from, double t_to,
                      HitRecord *out) override;

 private:
  __device__ bool CheckOnPlane(const Ray &ray, double t, double t_from,
                               double t_to, int axis);
};

template <bool HasTexCoord>
class FaceBase {
 protected:
  glm::vec3 positions_[3];
  glm::vec2 tex_coords_[3];

 public:
  __device__ __host__ inline glm::vec3 &position(int i) {
    return positions_[i];
  }
  __device__ __host__ inline glm::vec2 &tex_coord(int i) {
    return tex_coords_[i];
  }

  __device__ bool Hit(const Ray &ray, double t_from, double t_to,
                      HitRecord *out) {
    double u, v;
    bool ret = TriangleHit(positions_, ray, t_from, t_to, &out->t, &out->normal,
                           &u, &v);
    if (ret) {
      glm::vec2 tex_coord = tex_coords_[0] * (float)(1 - u - v) +
                            tex_coords_[1] * (float)u +
                            tex_coords_[2] * (float)v;
      out->u = tex_coord.x;
      out->v = tex_coord.y;
    }
    return ret;
  }
};

template <>
class FaceBase<false> {
 public:
  glm::vec3 positions_[3];

  __device__ __host__ inline glm::vec3 &position(int i) {
    return positions_[i];
  }

  __device__ bool Hit(const Ray &ray, double t_from, double t_to,
                      HitRecord *out) {
    double u, v;
    return TriangleHit(positions_, ray, t_from, t_to, &out->t, &out->normal, &u,
                       &v);
  }
};

template <bool HasTexCoord>
class Face : public FaceBase<HasTexCoord> {
 public:
  __device__ static AABB GetBV(Face<HasTexCoord> *objs, int n) {
    AABB aabb;
    aabb.min = glm::vec3(INFINITY);
    aabb.max = glm::vec3(-INFINITY);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) {
          aabb.min[k] = min(objs[i].positions_[j][k], aabb.min[k]);
          aabb.max[k] = max(objs[i].positions_[j][k], aabb.max[k]);
        }
    return aabb;
  }

  __device__ static int GetSplitAxis(Face<HasTexCoord> *objs, int n) {
    int split_axis = 0;
    auto aabb = Face<HasTexCoord>::GetBV(objs, n);
    glm::vec3 range = aabb.max - aabb.min;
    for (int i = 1; i <= 2; i++) {
      if (range[i] > range[split_axis]) {
        split_axis = i;
      }
    }
    return split_axis;
  }

  __device__ int operator<(const Face<HasTexCoord> &a) const {
    return this->positions_[0].x < a.positions_[0].x;
  }
};

template <typename T, typename BV>
class BVHNode : public Hitable {
 private:
  // tuned for 2080 ti
  constexpr static int kMin = 2048;

  T *objs_;
  int n_, mid_;
  BVHNode<T, BV> *left_ = nullptr, *right_ = nullptr;
  BV bv_;

 public:
  __device__ explicit BVHNode(T *objs, int n) : objs_(objs), n_(n) {
    bv_ = T::GetBV(objs_, n_);
    if (n_ <= kMin) return;
    int split_axis = T::GetSplitAxis(objs_, n_);
    thrust::sort(thrust::device, objs_, objs_ + n);
    mid_ = (n - 1) / 2;
    left_ = new BVHNode(objs_, mid_ + 1);
    right_ = new BVHNode(objs_ + mid_ + 1, n - mid_ - 1);
  }

  __device__ bool Hit(const Ray &ray, double t_from, double t_to,
                      HitRecord *out) override {
    if (left_ == nullptr) {
      bool ret = false;
      for (int i = 0; i < n_; i++) {
        HitRecord hit_record;
        if (objs_[i].Hit(ray, t_from, t_to, &hit_record)) {
          t_to = hit_record.t;
          *out = hit_record;
          ret = true;
        }
      }
      return ret;
    }

    bool left_hit = false, right_hit = false;
    HitRecord left_record, right_record;
    if (left_->bv_.Hit(ray, t_from, t_to, &left_record)) {
      left_hit = left_->Hit(ray, t_from, t_to, &left_record);
    }
    if (left_hit) {
      t_to = left_record.t;
    }
    if (right_->bv_.Hit(ray, t_from, t_to, &right_record)) {
      right_hit = right_->Hit(ray, t_from, t_to, &right_record);
    }
    if (!left_hit && !right_hit) {
      return false;
    }
    if (right_hit) {
      *out = right_record;
    } else {
      *out = left_record;
    }
    return true;
  }
};

template <typename T, typename BV>
class BVH : public Hitable {
 private:
  T *objs_;
  int n_;
  BVHNode<T, BV> *root_;
  Material *material_ptr_;

 public:
  __device__ explicit BVH(T *objs, int n, Material *material_ptr)
      : objs_(objs), n_(n), material_ptr_(material_ptr) {
    root_ = new BVHNode<T, BV>(objs_, n_);
  }

  __device__ bool Hit(const Ray &ray, double t_from, double t_to,
                      HitRecord *out) override {
    if (root_->Hit(ray, t_from, t_to, out)) {
      out->material_ptr = material_ptr_;
      return true;
    }
    return false;
  }
};