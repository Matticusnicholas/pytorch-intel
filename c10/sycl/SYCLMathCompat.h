// AI-TRANSLATED: SYCL math compatibility layer
// Maps CUDA math compat functions to SYCL equivalents
#pragma once

#include <cmath>

// SYCL TODO: needs_review - in production use:
// #include <sycl/sycl.hpp>
// and use sycl:: namespace math functions

namespace c10::sycl::compat {

template <typename T>
inline T tanh(T x) { return std::tanh(x); }

template <typename T>
inline T exp(T x) { return std::exp(x); }

template <typename T>
inline T log(T x) { return std::log(x); }

template <typename T>
inline T sqrt(T x) { return std::sqrt(x); }

template <typename T>
inline T abs(T x) { return std::abs(x); }

template <typename T>
inline T ceil(T x) { return std::ceil(x); }

template <typename T>
inline T floor(T x) { return std::floor(x); }

template <typename T>
inline T max(T a, T b) { return std::max(a, b); }

template <typename T>
inline T min(T a, T b) { return std::min(a, b); }

} // namespace c10::sycl::compat
