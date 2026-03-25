// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/Randperm.cuh
//
// SYCL: Random number generation uses oneMKL RNG or a SYCL-native
// Philox counter-based RNG instead of cuRAND.
#pragma once

#include <ATen/Utils.h>

namespace {

// SYCL: The randperm duplicate key handling kernel uses SYCL nd_range
// parallel_for with work-item indexing.
// curand_init/curand are replaced by a SYCL-compatible Philox RNG.

template<typename T, typename scalar_t>
void randperm_handle_duplicate_keys_sycl(
    T *keys, scalar_t *data, int bits, int64_t n,
    std::optional<at::Generator> &gen_);

// Alias for backward compatibility
template<typename T, typename scalar_t>
void randperm_handle_duplicate_keys(
    T *keys, scalar_t *data, int bits, int64_t n,
    std::optional<at::Generator> &gen_) {
  randperm_handle_duplicate_keys_sycl(keys, data, bits, n, gen_);
}

} // anonymous namespace
