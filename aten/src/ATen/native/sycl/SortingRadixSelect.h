// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/SortingRadixSelect.cuh
#pragma once

#include <ATen/ceil_div.h>
#include <c10/macros/Macros.h>
#include <type_traits>

namespace at::native {

template <typename scalar_t>
struct TopKTypeConfig {};

template <>
struct TopKTypeConfig<float> {
  typedef uint32_t RadixType;

  static inline RadixType convert(float v) {
    RadixType x;
    std::memcpy(&x, &v, sizeof(RadixType));
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    return (v == v) ? (x ^ mask) : 0xffffffff;
  }

  static inline float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
    RadixType x = v ^ mask;
    float result;
    std::memcpy(&result, &x, sizeof(float));
    return result;
  }
};

template <>
struct TopKTypeConfig<uint8_t> {
  typedef uint32_t RadixType;
  static inline RadixType convert(uint8_t v) { return v; }
  static inline uint8_t deconvert(RadixType v) { return v; }
};

template <>
struct TopKTypeConfig<int8_t> {
  typedef uint32_t RadixType;
  static inline RadixType convert(int8_t v) { return 128u + v; }
  static inline int8_t deconvert(RadixType v) { return v - 128; }
};

template <>
struct TopKTypeConfig<int16_t> {
  typedef uint32_t RadixType;
  static inline RadixType convert(int16_t v) { return 32768u + v; }
  static inline int16_t deconvert(RadixType v) { return v - 32768; }
};

template <>
struct TopKTypeConfig<int32_t> {
  typedef uint32_t RadixType;
  static inline RadixType convert(int32_t v) { return 2147483648u + v; }
  static inline int32_t deconvert(RadixType v) { return v - 2147483648u; }
};

template <>
struct TopKTypeConfig<int64_t> {
  typedef uint64_t RadixType;
  static inline RadixType convert(int64_t v) {
    return static_cast<uint64_t>(v) ^ (static_cast<uint64_t>(1) << 63);
  }
  static inline int64_t deconvert(RadixType v) {
    return static_cast<int64_t>(v ^ (static_cast<uint64_t>(1) << 63));
  }
};

template <>
struct TopKTypeConfig<double> {
  typedef uint64_t RadixType;

  static inline RadixType convert(double v) {
    RadixType x;
    std::memcpy(&x, &v, sizeof(RadixType));
    RadixType mask = (x >> 63) ? 0xffffffffffffffffULL : 0x8000000000000000ULL;
    return (v == v) ? (x ^ mask) : 0xffffffffffffffffULL;
  }

  static inline double deconvert(RadixType v) {
    RadixType mask = (v >> 63) ? 0x8000000000000000ULL : 0xffffffffffffffffULL;
    RadixType x = v ^ mask;
    double result;
    std::memcpy(&result, &x, sizeof(double));
    return result;
  }
};

template <>
struct TopKTypeConfig<at::Half> {
  typedef uint32_t RadixType;

  static inline RadixType convert(at::Half v) {
    RadixType x = v.x;
    RadixType mask = (x & 0x8000) ? 0xffff : 0x8000;
    return (v == v) ? (x ^ mask) : 0xffff;
  }

  static inline at::Half deconvert(RadixType v) {
    RadixType mask = (v & 0x8000) ? 0x8000 : 0xffff;
    at::Half result;
    result.x = static_cast<uint16_t>(v ^ mask);
    return result;
  }
};

template <>
struct TopKTypeConfig<at::BFloat16> {
  typedef uint32_t RadixType;

  static inline RadixType convert(at::BFloat16 v) {
    RadixType x = v.x;
    RadixType mask = (x & 0x8000) ? 0xffff : 0x8000;
    return (v == v) ? (x ^ mask) : 0xffff;
  }

  static inline at::BFloat16 deconvert(RadixType v) {
    RadixType mask = (v & 0x8000) ? 0x8000 : 0xffff;
    at::BFloat16 result;
    result.x = static_cast<uint16_t>(v ^ mask);
    return result;
  }
};

// SYCL: Radix select uses sycl::nd_item for indexing, sycl::group_barrier
// for synchronization, and sycl::sub_group for ballot/popcount operations
// instead of CUDA's __syncthreads(), __ballot_sync(), and __popc().

} // namespace at::native
