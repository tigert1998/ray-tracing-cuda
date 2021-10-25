#include "camera.cuh"
#include "utils.cuh"

using namespace glm;

Camera::Camera(vec3 position, vec3 look_at, vec3 up, double field_of_view,
               double width_height_aspect, double aperture,
               double focus_distance, curandState *state) {
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
  state_ = state;
}

Camera::Camera(vec3 position, vec3 look_at, vec3 up, double field_of_view,
               double width_height_aspect, curandState *state) {
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
  state_ = state;
}

Camera::Camera(vec3 position, vec3 lower_left_corner, vec3 horizontal,
               vec3 vertical, curandState *state) {
  is_defocus_camera_ = false;
  position_ = position;
  lower_left_corner_ = lower_left_corner;
  horizontal_ = horizontal;
  vertical_ = vertical;
  state_ = state;
}

vec3 Camera::position() const { return position_; }

vec3 Camera::lower_left_corner() const { return lower_left_corner_; }

vec3 Camera::horizontal() const { return horizontal_; }

vec3 Camera::vertical() const { return vertical_; }

Ray Camera::ray_at(double x, double y) {
  x = (x + 1) / 2;
  y = (y + 1) / 2;
  auto target = lower_left_corner() + static_cast<float>(x) * horizontal() +
                static_cast<float>(y) * vertical();
  vec3 origin;
  if (is_defocus_camera()) {
    vec2 offset = DiskRand(lens_radius_);
    origin = position() + u_ * offset.x + v_ * offset.y;
  } else {
    origin = position();
  }
  return Ray(origin, target - origin);
}

bool Camera::is_defocus_camera() const { return is_defocus_camera_; }

glm::vec2 Camera::DiskRand(float radius) {
  return vec2(cudaRandomFloat(0, radius, state_),
              cudaRandomFloat(0, radius, state_));
}
