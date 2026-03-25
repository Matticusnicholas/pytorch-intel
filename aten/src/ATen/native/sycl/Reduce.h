// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/Reduce.cuh
#pragma once

#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/sycl/MemoryAccess.h>
#include <ATen/native/sycl/KernelUtils.h>
#include <ATen/OpMathType.h>
#include <c10/macros/Macros.h>
#include <array>
#include <functional>
#include <iosfwd>
#include <type_traits>
#include <utility>

namespace at::native {

static inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

static inline int last_pow2(int n) {
  n |= (n >>  1);
  n |= (n >>  2);
  n |= (n >>  4);
  n |= (n >>  8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

C10_HOST_DEVICE static void reduce_fraction(size_t &numerator, size_t &denominator) {
  size_t a = denominator;
  size_t b = numerator;
  while (b != 0) {
      a %= b;
      size_t tmp = a;
      a = b;
      b = tmp;
  }
  numerator /= a;
  denominator /= a;
}

template <typename T>
struct mnt_wrapper {
  static constexpr int MAX_NUM_THREADS = 512;
};

template <>
struct mnt_wrapper <c10::complex<double>>{
  static constexpr int MAX_NUM_THREADS = 256;
};

constexpr int max_reduce_threads(c10::ScalarType type) {
  return type == kComplexDouble ? 256 : 512;
}

// SYCL sub-group size (analogous to CUDA warp size)
// Intel GPUs typically use sub-group sizes of 16 or 32
static constexpr int SYCL_SUB_GROUP_SIZE = 32;

struct ReduceConfig {
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  ReduceConfig(int element_size_bytes, int num_outputs, int num_inputs)
    : element_size_bytes(element_size_bytes)
    , num_inputs(num_inputs)
    , num_outputs(num_outputs) {}
  int element_size_bytes;
  int num_inputs;
  int num_outputs;
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};

  int block_width;
  int block_height;
  int num_threads;

  bool vectorize_input = false;
  int output_vec_size = 1;

  template <typename T>
  void set_block_dimension(int64_t dim0, int64_t dim1) {
    const int max_num_threads = mnt_wrapper<T>::MAX_NUM_THREADS / output_vec_size;
    int dim0_pow2 = dim0 < max_num_threads ? static_cast<int>(last_pow2(dim0)) : max_num_threads;
    int dim1_pow2 = dim1 < max_num_threads ? static_cast<int>(last_pow2(dim1)) : max_num_threads;
    block_width = std::min(dim0_pow2, SYCL_SUB_GROUP_SIZE);
    block_height = std::min(dim1_pow2, int(max_num_threads / block_width));
    block_width = std::min(dim0_pow2, int(max_num_threads / block_height));
    num_threads = block_width * block_height;
  }

  int split_input(int parallelism) {
    int step = step_input;
    step_input *= parallelism;
    return step;
  }

  int split_output(int parallelism) {
    int step = step_output;
    step_output *= parallelism;
    return step;
  }

  // SYCL: use sycl::range and sycl::nd_range instead of dim3
  sycl::range<2> local_range() const {
    return sycl::range<2>(block_height, block_width);
  }

  sycl::range<2> global_range() const {
    int grid_x = div_up(num_outputs / output_vec_size, step_output);
    int grid_y = ctas_per_output;
    return sycl::range<2>(grid_y * block_height, grid_x * block_width);
  }

  C10_HOST_DEVICE bool should_block_x_reduce() const {
    return input_mult[BLOCK_X] != 0;
  }

  C10_HOST_DEVICE bool should_block_y_reduce() const {
    return input_mult[BLOCK_Y] != 0;
  }

  C10_HOST_DEVICE bool should_global_reduce() const {
    return input_mult[CTA] != 0;
  }

  // SYCL: should_store uses item local_id instead of threadIdx
  bool should_store(int output_idx, sycl::nd_item<2> item) const {
    return output_idx < num_outputs &&
      (!should_block_x_reduce() || item.get_local_id(1) == 0) &&
      (!should_block_y_reduce() || item.get_local_id(0) == 0);
  }

  bool should_reduce_tail(sycl::nd_item<2> item) const {
    return (!should_block_y_reduce() || item.get_local_id(0) == 0) &&
      (!should_global_reduce() || item.get_group(0) == 0);
  }

  C10_HOST_DEVICE int input_idx(sycl::nd_item<2> item) const {
    int lane = item.get_local_id(1);  // threadIdx.x
    int warp = item.get_local_id(0);  // threadIdx.y
    int cta2 = item.get_group(0);     // blockIdx.y
    return (lane * input_mult[BLOCK_X] +
            warp * input_mult[BLOCK_Y] +
            cta2 * input_mult[CTA]);
  }

  template <int output_vec_size_t>
  C10_HOST_DEVICE int output_idx(sycl::nd_item<2> item) const {
    int lane = item.get_local_id(1);  // threadIdx.x
    int warp = item.get_local_id(0);  // threadIdx.y
    int cta1 = item.get_group(1);     // blockIdx.x
    return (lane * output_mult[BLOCK_X] +
            warp * output_mult[BLOCK_Y] +
            cta1 * step_output) * output_vec_size_t;
  }

  int shared_memory_offset(int offset, sycl::nd_item<2> item) const {
    return item.get_local_id(1) + (item.get_local_id(0) + offset) * block_width;
  }

  int staging_memory_offset(int cta2, sycl::nd_item<2> item) const {
    int offset = cta2 + item.get_group(1) * item.get_group_range(0);
    if (!should_block_x_reduce()) {
      offset = item.get_local_id(1) + offset * block_width;
    }
    return offset;
  }

  int shared_memory_size() const {
    if (!should_block_y_reduce() &&
        (!should_block_x_reduce() ||
         block_width <= SYCL_SUB_GROUP_SIZE)) {
      return 0;
    }
    return element_size_bytes * num_threads * output_vec_size;
  }

  int64_t global_memory_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    auto size = (int64_t)element_size_bytes * num_outputs * ctas_per_output;
    if (!should_block_x_reduce()) {
      size *= block_width * output_vec_size;
    }
    return size;
  }

  int semaphore_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    return sizeof(int) * (int)div_up(num_outputs / output_vec_size, step_output);
  }
};

// SYCL reduction kernel launch functions would be defined in a .cpp file
// using sycl::queue::submit with sycl::handler::parallel_for

// gpu_reduce_kernel equivalent for SYCL
template <typename scalar_t, typename out_scalar_t, int vt0, typename ops_t, typename ident_t>
void sycl_reduce_kernel(TensorIteratorBase& iter, const ops_t& ops, ident_t ident);

// Alias for backward compatibility
template <typename scalar_t, typename out_scalar_t, int vt0 = 4, typename ops_t, typename ident_t = double>
void gpu_reduce_kernel(TensorIteratorBase& iter, const ops_t& ops, ident_t ident = 0) {
  sycl_reduce_kernel<scalar_t, out_scalar_t, vt0>(iter, ops, ident);
}

} // namespace at::native
