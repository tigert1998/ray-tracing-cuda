#pragma once

#include <cuda_runtime.h>
#include <glog/logging.h>

class CudaCopyable {
 public:
  template <typename T>
  void ToDevice(T *symbol) {
    auto err = cudaMemcpy(symbol, this, sizeof(T), cudaMemcpyHostToDevice);
    CHECK(err == cudaSuccess) << cudaGetErrorString(err);
  }

  template <typename T>
  void FromDevice(T *symbol) {
    auto err = cudaMemcpy(this, symbol, sizeof(T), cudaMemcpyDeviceToHost);
    CHECK(err == cudaSuccess) << cudaGetErrorString(err);
  }
};