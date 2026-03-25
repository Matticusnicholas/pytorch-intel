// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/GroupMMCommon.cuh
//
// SYCL: CUTLASS is not available for SYCL. Grouped GEMM operations
// on Intel GPUs should use oneMKL or oneDNN for batched/grouped GEMM.
#pragma once

#include <array>
#include <cstdint>

namespace at::sycl::detail {

using Strides = std::array<int64_t, 3>;

// SYCL: The CUTLASS-based grouped GEMM data preparation kernel is not
// available for SYCL. Use oneMKL batch GEMM APIs instead.
// This header is provided for interface compatibility.

} // namespace at::sycl::detail
