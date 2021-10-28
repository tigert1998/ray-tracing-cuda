#include "bvh.cuh"
#include "utils.cuh"

using glm::vec3;

__device__ bool BB::CheckOnPlane(const Ray &ray, double t, double t_from,
                                 double t_to, int axis) {
  if (std::isnan(t) || std::isinf(t)) return false;
  if (!(t_from <= t && t <= t_to)) return false;
  vec3 pt_on_plane = ray.position() + (float)t * ray.direction();
  for (int i = 0; i < 3; i++) {
    if (i == axis) continue;
    if (min[i] <= pt_on_plane[i] && pt_on_plane[i] <= max[i]) continue;
    return false;
  }
  return true;
}

__device__ bool BB::Hit(const Ray &ray, double t_from, double t_to,
                        HitRecord *out) {
  vec3 dir = ray.direction(), pos = ray.position();

  for (int i = 0; i < 3; i++) {
    if (dir[i] == 0.f) continue;
    double ts[] = {(min[i] - pos[i]) / dir[i], (max[i] - pos[i]) / dir[i]};
    if (CheckOnPlane(ray, ts[0], t_from, t_to, i)) return true;
    if (CheckOnPlane(ray, ts[1], t_from, t_to, i)) return true;
  }
  return false;
}

__device__ BVHNode::BVHNode(Face *faces, int n) : faces_(faces), n_(n) {
  bb_.min = vec3(INFINITY);
  bb_.max = vec3(-INFINITY);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++) {
        bb_.min[k] = min(faces_[i].points[j][k], bb_.min[k]);
        bb_.max[k] = max(faces_[i].points[j][k], bb_.max[k]);
      }
  if (n_ <= kMinFaces) return;

  int split_axis = 0;
  glm::vec3 range = bb_.max - bb_.min;
  for (int i = 1; i <= 2; i++) {
    if (range[i] > range[split_axis]) {
      split_axis = i;
    }
  }

  nvstd::function<int(const Face &, const Face &)> comp =
      [split_axis](const Face &a, const Face &b) {
        return a.points[0][split_axis] - b.points[0][split_axis];
      };
  QuickSort<Face>(faces_, 0, n - 1, comp);
  mid_ = (n - 1) / 2;
  left_ = new BVHNode(faces, mid_ + 1);
  right_ = new BVHNode(faces + mid_ + 1, n - mid_ - 1);
}

__device__ bool BVHNode::Hit(const Ray &ray, double t_from, double t_to,
                             HitRecord *out) {
  if (left_ == nullptr) {
    bool ret = false;
    for (int i = 0; i < n_; i++) {
      double t;
      glm::vec3 normal;
      if (TriangleHit(faces_[i].points, ray, t_from, t_to, &t, &normal)) {
        t_to = t;
        out->t = t;
        out->normal = normal;
        ret = true;
      }
    }
    return ret;
  }

  bool left_hit = false, right_hit = false;
  HitRecord left_record, right_record;
  if (left_->bb_.Hit(ray, t_from, t_to, &left_record)) {
    left_hit = left_->Hit(ray, t_from, t_to, &left_record);
  }
  if (left_hit) {
    t_to = left_record.t;
  }
  if (right_->bb_.Hit(ray, t_from, t_to, &right_record)) {
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

__device__ BVH::BVH(Face *faces, int n, Material *material_ptr)
    : faces_(faces), n_(n), material_ptr_(material_ptr) {
  root_ = new BVHNode(faces_, n_);
}

__device__ bool BVH::Hit(const Ray &ray, double t_from, double t_to,
                         HitRecord *out) {
  if (root_->Hit(ray, t_from, t_to, out)) {
    out->material_ptr = material_ptr_;
    return true;
  }
  return false;
}