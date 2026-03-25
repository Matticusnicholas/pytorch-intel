// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/Pow.cuh
#pragma once
#include <ATen/native/Pow.h>
#include <c10/core/Scalar.h>
#include <cmath>
#include <type_traits>

namespace at::native {

namespace {

// SYCL: No __host__ __device__ annotations needed; these are compiled
// for both host and device by the DPC++ compiler.
static inline at::Half pow_(at::Half base, at::Half exp) {
  return static_cast<at::Half>(std::pow(static_cast<float>(base), static_cast<float>(exp)));
}

static inline at::BFloat16 pow_(at::BFloat16 base, at::BFloat16 exp) {
  return static_cast<at::BFloat16>(std::pow(static_cast<float>(base), static_cast<float>(exp)));
}

template <typename Base_type, typename Exp_type>
static inline typename std::enable_if_t<std::is_floating_point_v<Base_type> && (std::is_same_v<Base_type, Exp_type> || std::is_same_v<Exp_type, int>), Base_type>
  pow_(Base_type base, Exp_type exp) {
  return std::pow(base, exp);
}

template <typename Base_type, typename Exp_type>
static inline typename std::enable_if_t<!std::is_same_v<Base_type, Exp_type> && !std::is_same_v<Exp_type, int>, Base_type>
  pow_(Base_type base, Exp_type exp) {
  return static_cast<Base_type>(std::pow(static_cast<double>(base), static_cast<double>(exp)));
}

template <typename T>
static inline std::enable_if_t<std::is_integral_v<T>, T> pow_(T base, T exp) {
  return at::native::powi(base, exp);
}

template <typename T>
static inline c10::complex<T> pow_(c10::complex<T> base, c10::complex<T> exp) {
  return c10_complex_math::pow(base, exp);
}

} // namespace
} // namespace at::native
