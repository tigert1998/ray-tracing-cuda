#include "bvh.cuh"
#include "utils.cuh"

__device__ bool BV::Hit(const Ray &ray, double t_from, double t_to,
                        HitRecord *out) {
  // TODO
  return true;
}

__device__ BVHNode::BVHNode(Face *faces, int n) : faces_(faces), n_(n) {
  bv_.x[0] = bv_.y[0] = bv_.z[0] = INFINITY;
  bv_.x[1] = bv_.y[1] = bv_.z[1] = -INFINITY;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 3; j++) {
      bv_.x[0] = min(bv_.x[0], faces_[i].points[j].x);
      bv_.x[1] = max(bv_.x[1], faces_[i].points[j].x);
      bv_.y[0] = min(bv_.y[0], faces_[i].points[j].y);
      bv_.y[1] = max(bv_.y[1], faces_[i].points[j].y);
      bv_.z[0] = min(bv_.z[0], faces_[i].points[j].z);
      bv_.z[1] = max(bv_.z[1], faces_[i].points[j].z);
    }
  if (n_ <= 1) return;
  int mod = n_ % 3;
  nvstd::function<int(const Face &, const Face &)> comp = nullptr;
  if (mod == 0) {
    comp = [](const Face &a, const Face &b) {
      return a.points[0].x - b.points[0].x;
    };
  } else if (mod == 1) {
    comp = [](const Face &a, const Face &b) {
      return a.points[0].y - b.points[0].y;
    };
  } else {
    comp = [](const Face &a, const Face &b) {
      return a.points[0].z - b.points[0].z;
    };
  }
  QuickSort<Face>(faces_, 0, n - 1, comp);
  mid_ = (n - 1) / 2;
  left_ = new BVHNode(faces, mid_ + 1);
  right_ = new BVHNode(faces + mid_ + 1, n - mid_ - 1);
}

__device__ bool BVHNode::Hit(const Ray &ray, double t_from, double t_to,
                             HitRecord *out) {
  if (n_ == 1) {
    // TODO
    return true;
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

__device__ BVH::BVH(Face *faces, int n) : faces_(faces), n_(n) {
  root_ = new BVHNode(faces_, n_);
}

__device__ bool BVH::Hit(const Ray &ray, double t_from, double t_to,
                         HitRecord *out) {
  return root_->Hit(ray, t_from, t_to, out);
}