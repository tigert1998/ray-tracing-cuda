#include "hitable_list.cuh"

using std::pair;

#include <iostream>

__device__ bool HitableList::Hit(const Ray &ray, double t_from, double t_to,
                                 HitRecord *out) {
  HitRecord hit_record;
  bool ok = false;
  for (int i = 0; i < list_len_; i++) {
    auto ret = list_[i]->Hit(ray, t_from, t_to, &hit_record);
    if (!ret) continue;
    if (!ok) {
      *out = hit_record;
      ok = true;
    } else {
      if (out->t > hit_record.t) {
        *out = hit_record;
      }
    }
  }
  return ok;
}

__host__ __device__ void HitableList::Append(Hitable *obj) {
  list_[list_len_++] = obj;
}
