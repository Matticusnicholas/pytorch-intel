// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/UniqueCub.cuh
//
// SYCL: cub::DeviceSelect::Unique is replaced by oneDPL unique operations
// #include <oneapi/dpl/algorithm>
// #include <oneapi/dpl/execution>
#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::internal {

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_sycl_template(
    const Tensor& self,
    const bool consecutive,
    const bool return_inverse,
    const bool return_counts);

// Alias for backward compatibility
template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_cuda_template(
    const Tensor& self,
    const bool consecutive,
    const bool return_inverse,
    const bool return_counts) {
  return unique_sycl_template<scalar_t>(self, consecutive, return_inverse, return_counts);
}

} // namespace at::native::internal
