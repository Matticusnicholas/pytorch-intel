// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/KernelUtils.cuh
#pragma once

#include <c10/macros/Macros.h>
#include <type_traits>

namespace at::native {

// SYCL index helper: replaces CUDA's (nc * height + h) * width + w
inline size_t
idx(const size_t nc,
    const size_t height,
    const size_t width,
    const size_t h,
    const size_t w) {
  return (nc * height + h) * width + w;
}

// for channels-last
inline size_t
idx_cl(
  const size_t n, const size_t h, const size_t w, const size_t c,
  const size_t height, const size_t width, const size_t channel
) {
  return ((n * height + h) * width + w) * channel + c;
}

// SYCL atomic add using sycl::atomic_ref
// Replaces CUDA's fastSpecializedAtomicAdd / fastAtomicAdd
template <typename scalar_t, typename index_t>
inline void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    index_t index,
    const index_t numel,
    scalar_t value) {
  // SYCL: Use sycl::atomic_ref for atomic operations
  // For half/bfloat16, the SYCL runtime handles the appropriate atomic
  // operation based on device capabilities
  sycl::atomic_ref<scalar_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space> atomic_val(tensor[index]);
  atomic_val.fetch_add(value);
}

// Specialization for float
template <typename index_t>
inline void fastSpecializedAtomicAdd(
    float* tensor,
    index_t index,
    const index_t numel,
    float value) {
  sycl::atomic_ref<float,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space> atomic_val(tensor[index]);
  atomic_val.fetch_add(value);
}

// Specialization for double
template <typename index_t>
inline void fastSpecializedAtomicAdd(
    double* tensor,
    index_t index,
    const index_t numel,
    double value) {
  sycl::atomic_ref<double,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space> atomic_val(tensor[index]);
  atomic_val.fetch_add(value);
}

template <class scalar_t, class index_t>
inline void fastAtomicAdd(
    scalar_t* tensor,
    index_t index,
    const index_t numel,
    scalar_t value,
    bool fast_atomics) {
  // SYCL: sycl::atomic_ref handles all cases
  fastSpecializedAtomicAdd(tensor, index, numel, value);
}

} // namespace at::native
