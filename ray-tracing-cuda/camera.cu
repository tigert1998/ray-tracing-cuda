#include "camera.cuh"
#include "utils.cuh"

using namespace glm;

__host__ __device__ Camera::Camera(vec3 position, vec3 look_at, vec3 up,
                                   double field_of_view,
                                   double width_height_aspect, double aperture,
                                   double focus_distance) {
  is_defocus_camera_ = true;
  position_ = position;
  w_ = normalize(position - look_at);
  u_ = normalize(cross(up, w_));
  v_ = normalize(cross(w_, u_));
  double half_height = focus_distance * tan(field_of_view / 2);
  double half_width = width_height_aspect * half_height;
  horizontal_ = u_ * static_cast<float>(2 * half_width);
  vertical_ = v_ * static_cast<float>(2 * half_height);
  lower_left_corner_ = position - w_ - u_ * static_cast<float>(half_width) -
                       v_ * static_cast<float>(half_height);
  lens_radius_ = aperture / 2;
}

__host__ __device__ Camera::Camera(vec3 position, vec3 look_at, vec3 up,
                                   double field_of_view,
                                   double width_height_aspect) {
  is_defocus_camera_ = false;
  position_ = position;
  w_ = normalize(position - look_at);
  u_ = normalize(cross(up, w_));
  v_ = normalize(cross(w_, u_));
  double half_height = tan(field_of_view / 2);
  double half_width = width_height_aspect * half_height;
  horizontal_ = u_ * static_cast<float>(2 * half_width);
  vertical_ = v_ * static_cast<float>(2 * half_height);
  lower_left_corner_ = position - w_ - u_ * static_cast<float>(half_width) -
                       v_ * static_cast<float>(half_height);
}

__host__ __device__ Camera::Camera(vec3 position, vec3 lower_left_corner,
                                   vec3 horizontal, vec3 vertical) {
  is_defocus_camera_ = false;
  position_ = position;
  lower_left_corner_ = lower_left_corner;
  horizontal_ = horizontal;
  vertical_ = vertical;
}

__device__ vec3 Camera::position() const { return position_; }

__device__ vec3 Camera::lower_left_corner() const { return lower_left_corner_; }

__device__ vec3 Camera::horizontal() const { return horizontal_; }

__device__ vec3 Camera::vertical() const { return vertical_; }

__device__ Ray Camera::RayAt(double x, double y, curandState *state) {
  x = (x + 1) / 2;
  y = (y + 1) / 2;
  auto target = lower_left_corner() + static_cast<float>(x) * horizontal() +
                static_cast<float>(y) * vertical();
  vec3 origin;
  if (is_defocus_camera()) {
    vec2 offset = DiskRand(lens_radius_, state);
    origin = position() + u_ * offset.x + v_ * offset.y;
  } else {
    origin = position();
  }
  return Ray(origin, target - origin);
}

__device__ bool Camera::is_defocus_camera() const { return is_defocus_camera_; }

__device__ glm::vec2 Camera::DiskRand(float radius, curandState *state) {
  return vec2(CudaRandomFloat(0, radius, state),
              CudaRandomFloat(0, radius, state));
}
