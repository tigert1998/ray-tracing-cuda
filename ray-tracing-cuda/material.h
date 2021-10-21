#pragma once

class Material;

#include <glm/glm.hpp>

#include "hitable.h"
#include "ray.h"

class Material {
 public:
  virtual bool Scatter(const Ray &ray, const HitRecord &record,
                       std::pair<glm::vec3, Ray> *out) const = 0;
  virtual glm::vec3 Emit(double u, double v, const glm::vec3 &p) const;
};
