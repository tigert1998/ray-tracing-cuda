#pragma once

#include <utility>

#include "hitable.h"
#include "material.h"

class Sphere : public Hitable {
 public:
  Sphere() = delete;
  Sphere(glm::vec3 position, double radius, Material* material_ptr);
  bool Hit(const Ray& ray, std::pair<double, double> t_range,
           HitRecord* out) const;
  double radius() const;
  glm::vec3 position() const;
  Material* material_ptr() const;

 private:
  double radius_;
  glm::vec3 position_;
  Material* material_ptr_;
};
