// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/block_reduce.cuh
#pragma once

#include <ATen/native/SharedReduceOps.h>
#include <c10/macros/Macros.h>
#include <limits>

namespace at::native::sycl_utils {

// SYCL sub-group size (replaces C10_WARP_SIZE)
// Intel GPUs typically use sub-group sizes of 16 or 32
static constexpr int kSYCLSubGroupSize = 32;

constexpr int kSYCLBlockReduceNumThreads = 512;

constexpr int kSYCLBlockReduceMaxThreads() {
    return kSYCLSubGroupSize * kSYCLSubGroupSize;
}

// Sums val across all work-items in a sub-group (replaces WarpReduceSum).
// SYCL uses sycl::sub_group for sub-group operations.
template <typename T>
inline T SubGroupReduceSum(T val, sycl::sub_group sg) {
  return sycl::reduce_over_group(sg, val, sycl::plus<T>());
}

// Alternative implementation using explicit shuffles (more direct translation)
template <typename T>
inline T SubGroupReduceSumShuffle(T val, sycl::sub_group sg) {
  auto sg_size = sg.get_local_range()[0];
  for (unsigned int offset = sg_size / 2; offset > 0; offset >>= 1) {
    val += sycl::shift_group_left(sg, val, offset);
  }
  return val;
}

// Max across sub-group (replaces WarpReduceMax)
template <typename T>
inline T SubGroupReduceMax(T val, sycl::sub_group sg) {
  auto sg_size = sg.get_local_range()[0];
  for (unsigned int offset = sg_size / 2; offset > 0; offset >>= 1) {
    T other = sycl::shift_group_left(sg, val, offset);
    val = max_propagate_nan(val, other);
  }
  return val;
}

struct Block1D {
    static inline int Tid(sycl::nd_item<1> item) { return item.get_local_id(0); }
    static inline int SubGroups(sycl::nd_item<1> item) {
        return item.get_local_range(0) / kSYCLSubGroupSize;
    }
};

struct Block2D {
    static inline int Tid(sycl::nd_item<2> item) {
        return item.get_local_id(1) + item.get_local_id(0) * item.get_local_range(1);
    }
    static inline int SubGroups(sycl::nd_item<2> item) {
        return item.get_local_range(0) * item.get_local_range(1) / kSYCLSubGroupSize;
    }
};

// Block-level sum reduction via sub-group reduce + shared memory.
// Warning: return value only valid for work-item 0.
// shared: pointer to local memory with size >= sizeof(T) * num_sub_groups
template <typename T, typename B = Block1D, typename ItemT>
inline T BlockReduceSum(T val, T* shared, ItemT item, sycl::sub_group sg) {
  const int tid = B::Tid(item);
  const int lid = tid % kSYCLSubGroupSize;
  const int wid = tid / kSYCLSubGroupSize;

  val = SubGroupReduceSumShuffle(val, sg);
  // SYCL: item.barrier(sycl::access::fence_space::local_space) replaces __syncthreads()
  sycl::group_barrier(item.get_group());
  if (lid == 0) {
    shared[wid] = val;
  }
  sycl::group_barrier(item.get_group());
  val = (tid < B::SubGroups(item)) ? shared[lid] : T(0);
  if (wid == 0) {
    val = SubGroupReduceSumShuffle(val, sg);
  }
  return val;
}

// Block-level max reduction.
// Warning: return value only valid for work-item 0.
template <typename T, typename B = Block1D, typename ItemT>
inline T BlockReduceMax(T val, T* shared, ItemT item, sycl::sub_group sg) {
  const int tid = B::Tid(item);
  const int lid = tid % kSYCLSubGroupSize;
  const int wid = tid / kSYCLSubGroupSize;

  val = SubGroupReduceMax(val, sg);
  sycl::group_barrier(item.get_group());
  if (lid == 0) {
    shared[wid] = val;
  }
  sycl::group_barrier(item.get_group());
  val = (tid < B::SubGroups(item)) ? shared[lid] : T(std::numeric_limits<T>::lowest());
  if (wid == 0) {
    val = SubGroupReduceMax(val, sg);
  }
  return val;
}

// Generic sub-group reduce with user-defined op
template <typename T, class ReduceOp>
inline T SubGroupReduce(T val, const ReduceOp& op, sycl::sub_group sg) {
  auto sg_size = sg.get_local_range()[0];
  for (unsigned int offset = sg_size / 2; offset > 0; offset >>= 1) {
    val = op.combine(val, sycl::shift_group_left(sg, val, offset));
  }
  return val;
}

// Generic block reduce with user-defined op
template <typename T, class ReduceOp, typename B = Block1D, typename ItemT>
inline T BlockReduce(T val, const ReduceOp& op, const T& identity_element, T* shared, ItemT item, sycl::sub_group sg) {
  const int tid = B::Tid(item);
  const int lid = tid % kSYCLSubGroupSize;
  const int wid = tid / kSYCLSubGroupSize;

  val = SubGroupReduce(val, op, sg);
  sycl::group_barrier(item.get_group());
  if (lid == 0) {
    shared[wid] = val;
  }
  sycl::group_barrier(item.get_group());
  val = (tid < B::SubGroups(item)) ? shared[lid] : identity_element;
  if (wid == 0) {
    val = SubGroupReduce(val, op, sg);
  }
  return val;
}

// Backward-compatible aliases using CUDA naming
namespace cuda_utils = sycl_utils;

} // namespace at::native::sycl_utils

// Provide backward-compatible namespace alias
namespace at::native::cuda_utils {
  using namespace at::native::sycl_utils;
}
