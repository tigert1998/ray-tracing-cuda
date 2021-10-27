#include "parallelepiped.cuh"
#include "parallelogram.cuh"

using glm::vec3;
using std::function;
using std::pair;

__host__ __device__ Parallelepiped::Parallelepiped(glm::vec3 p[4],
                                                   Material *material_ptr) {
  auto fourth = [](vec3 a, vec3 b, vec3 c) -> vec3 { return c + b - a; };
  vec3 q[4];
  q[3] = fourth(p[0], p[1], p[2]);
  q[2] = fourth(p[0], p[1], p[3]);
  q[1] = fourth(p[0], p[2], p[3]);
  q[0] = fourth(p[1], q[2], q[3]);
  AddCorner(p, material_ptr);
  AddCorner(q, material_ptr);
}

__device__ bool Parallelepiped::Hit(const Ray &ray, double t_from, double t_to,
                                    HitRecord *out) {
  return list_.Hit(ray, t_from, t_to, out);
}

__device__ __host__ void Parallelepiped::AddCorner(glm::vec3 p[4],
                                                   Material *material_ptr) {
  for (int i = 1; i <= 3; i++) {
    int x = i, y = i + 1 == 4 ? 1 : x + 1;
    vec3 arr[3] = {p[0], p[x], p[y]};
    list_.Append(new Parallelogram(arr, material_ptr));
  }
};

__host__ __device__ Parallelepiped::Parallelepiped(
    glm::vec3 lengths, Material *material_ptr,
    nvstd::function<glm::vec3(glm::vec3)> transform) {
  vec3 p[4];
  vec3 q[4];
  p[0] = vec3(0);
  for (int i = 1; i <= 3; i++) {
    p[i] = vec3(0);
    p[i][i - 1] = lengths[i - 1];
  }
  q[0] = lengths;
  for (int i = 1; i <= 3; i++) {
    q[i] = lengths;
    q[i][i - 1] = 0;
  }
  for (int i = 0; i < 4; i++) {
    p[i] = transform(p[i]);
    q[i] = transform(q[i]);
  }
  AddCorner(p, material_ptr);
  AddCorner(q, material_ptr);
}