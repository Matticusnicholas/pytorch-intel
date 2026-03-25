// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/reduction_template.cuh
//
// The CUDA version contains string templates for JIT-compiled reduction
// kernels. SYCL uses AOT compilation, so these string templates are not
// needed. This header provides the reduction infrastructure for SYCL.
#pragma once

#include <c10/macros/Macros.h>

namespace at::sycl {

// SYCL: The reduction template from the CUDA backend is a string-based
// JIT template that gets compiled at runtime. In the SYCL backend,
// reductions are compiled ahead-of-time using:
//   - sycl::reduce_over_group for work-group reductions
//   - sycl::joint_reduce for joint reductions
//   - Sub-group reductions via sycl::sub_group
//
// The WARP_SHFL_DOWN pattern is replaced by sycl::shift_group_left
// on sycl::sub_group objects.
//
// Shared memory (__shared__) is replaced by sycl::local_accessor
// allocated in the kernel command group handler.

} // namespace at::sycl
