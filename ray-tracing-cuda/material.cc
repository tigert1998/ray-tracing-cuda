#include "material.h"

__device__ glm::vec3 Material::Emit(double u, double v,
                                    const glm::vec3 &p) const {
  return glm::vec3(0);
}