// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/DeviceSqrt.cuh
#pragma once

#include <cmath>

namespace at::native {

// SYCL: std::sqrt works for all types in SYCL device code
template<typename scalar_t>
inline double device_sqrt(scalar_t val) {
  return std::sqrt(static_cast<double>(val));
}

template<>
inline double device_sqrt<float>(float val) {
  return std::sqrt(val);
}

template<>
inline double device_sqrt<double>(double val) {
  return std::sqrt(val);
}

} // namespace at::native
