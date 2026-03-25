// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/cutlass_common.cuh
//
// SYCL: CUTLASS is not available for SYCL backends.
// Intel GPU GEMM operations should use oneMKL or oneDNN instead.
// This header provides stubs for interface compatibility.
#pragma once

#include <c10/util/Exception.h>

namespace at::sycl::detail {

// SYCL: CUTLASS kernel wrappers are not applicable to SYCL.
// For GEMM operations on Intel GPUs, use:
//   - oneMKL (oneapi::mkl::blas::gemm)
//   - oneDNN (dnnl::matmul)
//
// The enable_*_kernel_for_sm* structs from the CUDA version are
// architecture-specific optimizations that do not apply to Intel GPUs.

} // namespace at::sycl::detail
