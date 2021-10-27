#include <glog/logging.h>

#include <fstream>
#include <string>

#include "utils.cuh"

__global__ void CudaRandomInit(uint64_t seed, curandState *state) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}

void WriteImage(std::vector<glm::vec3> &pixels, int height, int width) {
  std::string title = "image.ppm";
  LOG(INFO) << "Writing to " << title << "..." << std::endl;
  std::fstream fs(title, std::ios::out);
  fs << "P3\n" << width << " " << height << "\n255\n";
  for (int i = 0; i < width * height; i++) {
    auto color = pixels[i];
    for (int j = 0; j < 3; j++) fs << int(color[j] * 255) << " ";
  }
}