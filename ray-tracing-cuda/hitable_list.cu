#include "hitable_list.cuh"

using std::pair;

#include <iostream>

__device__ bool HitableList::Hit(const Ray &ray, double t_from, double t_to,
                                 HitRecord *out) {
  bool ok = false;
  HitRecord hit_record;
  for (int i = 0; i < list_len_; i++) {
    bool ret = list_[i]->Hit(ray, t_from, t_to, &hit_record);
    if (ret) {
      if (!ok) {
        *out = hit_record;
        t_to = min(t_to, hit_record.t);
        ok = true;
      } else if (hit_record.t < t_to) {
        t_to = hit_record.t;
        *out = hit_record;
      }
    }
  }
  return ok;
}

__host__ __device__ void HitableList::Append(Hitable *obj) {
  list_[list_len_++] = obj;
}
