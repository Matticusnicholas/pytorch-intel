// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/ScanUtils.cuh
#pragma once
#include <ATen/NumericUtils.h>
#include <ATen/core/TensorBase.h>
#include <c10/util/Load.h>
#include <limits>
#include <cmath>

namespace at::native {

template <typename integer>
constexpr inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

template <typename integer>
constexpr inline integer get_log_num_threads_x_inner_scan(integer num_rows, integer row_size) {
  integer log_num_threads_x = 0;
  integer log_num_threads_y = 0;
  while (((integer)1 << log_num_threads_x) < row_size) {
    ++log_num_threads_x;
  }
  while (((integer)1 << log_num_threads_y) < num_rows) {
    ++log_num_threads_y;
  }
  integer diff = log_num_threads_x - log_num_threads_y;
  log_num_threads_x = ((integer)9 + diff) / (integer)2;
  log_num_threads_x = std::min(std::max((integer)4, log_num_threads_x), (integer)9);
  return log_num_threads_x;
}

template<typename scalar_t, typename idx_t, typename BinaryOperation>
void binary_op_update(const scalar_t lhs, scalar_t& rhs, const idx_t lhs_idx, idx_t& rhs_idx, BinaryOperation binary_op) {
  if(!at::_isnan(rhs) && (at::_isnan(lhs) || !binary_op(rhs, lhs))) {
    rhs = lhs;
    rhs_idx = lhs_idx;
  }
}

// SYCL scan kernels use sycl::nd_item for work-item indexing
// and sycl::group_barrier for synchronization instead of __syncthreads()
// oneDPL provides scan primitives that can also be used.

} // namespace at::native
