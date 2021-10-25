#include "hitable_list.cuh"

using std::pair;

__device__ bool HitableList::Hit(const Ray &ray,
                                 std::pair<double, double> t_range,
                                 HitRecord *out) const {
  HitRecord hit_record;
  bool ok = false;
  for (int i = 0; i < list_len_; i++) {
    auto ret = list_[i]->Hit(ray, t_range, &hit_record);
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

void HitableList::Append(Hitable *obj) { list_[list_len_++] = obj; }
