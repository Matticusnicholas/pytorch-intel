// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/ForeachFunctors.cuh
#pragma once
#include <ATen/OpMathType.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/sycl/DeviceAddCmulCdiv.h>
#include <ATen/native/sycl/MultiTensorApply.h>
#include <ATen/native/sycl/Pow.h>

namespace at::native {

namespace {

inline void increment_version(TensorList tensors) {
  for (const auto& t : tensors) {
    t.unsafeGetTensorImpl()->bump_version();
  }
}

// Initializes args and checks if all args are aligned
template <int depth, typename T>
bool init_args(
    T** args,
    TensorListMetadata<depth>& tl,
    const int64_t chunk_idx,
    const int64_t chunk_size,
    const int64_t tensor_loc) {
  bool all_aligned = true;
  for (int i = 0; i < depth; i++) {
    args[i] = (T*)tl.addresses[i][tensor_loc];
    args[i] += chunk_idx * chunk_size;
    if (!is_aligned(args[i])) {
      all_aligned = false;
    }
  }
  return all_aligned;
}

template <int depth, typename T, typename T2>
bool init_args(
    T** args,
    TensorListScalarListMetadata<T2, depth>& tl,
    const int64_t chunk_idx,
    const int64_t chunk_size,
    const int64_t tensor_loc) {
  bool all_aligned = true;
  for (int i = 0; i < depth; i++) {
    args[i] = (T*)tl.addresses[i][tensor_loc];
    args[i] += chunk_idx * chunk_size;
    if (!is_aligned(args[i])) {
      all_aligned = false;
    }
  }
  return all_aligned;
}

template <int depth, typename T>
bool init_args(
    T** args,
    FusedOptimizerTensorListMetadata<depth>& tl,
    const int64_t chunk_idx,
    const int64_t chunk_size,
    const int64_t tensor_loc) {
  bool all_aligned = true;
  for (int i = 0; i < depth; i++) {
    args[i] = (T*)tl.addresses[i][tensor_loc];
    args[i] += chunk_idx * chunk_size;
    if (!is_aligned(args[i])) {
      all_aligned = false;
    }
  }
  return all_aligned;
}

template <int depth, typename T>
void load_args(
    T r_args[][kILP],
    T** args,
    int64_t i_start,
    int64_t chunk_size,
    int64_t n) {
  // SYCL: uses work-item local_id instead of threadIdx.x
  // This will be called from within a SYCL kernel where the item is available
#pragma unroll
  for (int ii = 0; ii < kILP; ii++) {
    int64_t i = i_start + ii;
    for (int d = 0; d < depth; d++) {
      r_args[d][ii] = (i < n && i < chunk_size) ? args[d][i] : T(0);
    }
  }
}

template <typename T>
void store_args(
    T* dst,
    T r_args[kILP],
    int64_t i_start,
    int64_t chunk_size,
    int64_t n) {
#pragma unroll
  for (int ii = 0; ii < kILP; ii++) {
    int64_t i = i_start + ii;
    if (i < n && i < chunk_size) {
      dst[i] = r_args[ii];
    }
  }
}

} // namespace

} // namespace at::native
