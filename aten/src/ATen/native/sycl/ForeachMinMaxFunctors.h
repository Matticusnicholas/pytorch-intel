// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/ForeachMinMaxFunctors.cuh
#pragma once

#include <ATen/NumericUtils.h>

namespace at::native {

template <typename T>
struct minimum {
  T operator()(const T& a, const T& b) const {
    return (_isnan(a) || a < b) ? a : b;
  }
};

template <typename T>
struct maximum {
  T operator()(const T& a, const T& b) const {
    return (_isnan(a) || a > b) ? a : b;
  }
};

} // namespace at::native
