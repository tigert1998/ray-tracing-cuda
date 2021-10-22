#pragma once

#include <memory>
#include <vector>

#include "hitable.cuh"

class HitableList : public Hitable {
 public:
  const static int kMaxHitables = 32;

  __device__ bool Hit(const Ray &ray, std::pair<double, double> t_range,
                      HitRecord *out) const override;

  void Append(Hitable *obj);

 private:
  Hitable *list_[kMaxHitables];
  int list_len_ = 0;
};
