#pragma once

#include <memory>
#include <vector>

#include "hitable.cuh"

class HitableList : public Hitable {
 public:
  constexpr static int kMaxHitables = 32;

  __device__ bool Hit(const Ray &ray, std::pair<double, double> t_range,
                      HitRecord *out) override;

  __host__ __device__ void Append(Hitable *obj);

  __inline__ __host__ __device__ int list_len() const { return list_len_; }

 private:
  Hitable *list_[kMaxHitables];
  int list_len_ = 0;
};
