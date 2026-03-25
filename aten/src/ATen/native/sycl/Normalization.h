// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/Normalization.cuh
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/native/sycl/BlockReduce.h>
#include <ATen/native/sycl/DeviceSqrt.h>
#include <ATen/native/sycl/KernelUtils.h>
#include <c10/macros/Macros.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

// SYCL: Max block size for Intel GPUs
constexpr int SYCL_MAX_BLOCK_SIZE = 512;
constexpr unsigned SYCL_MAX_GRID_SIZE = 65535u;

static int getNumThreads(int nElem) {
  int threadSizes[5] = { 32, 64, 128, 256, SYCL_MAX_BLOCK_SIZE };
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return SYCL_MAX_BLOCK_SIZE;
}

// SYCL: Uses sycl::clz instead of __clz
inline int getMSB(int val) {
  return 31 - sycl::clz(val);
}

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  Float2() {}
  Float2(scalar_t v1, scalar_t v2) : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  Float2(int v) : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
  friend Float2 operator+(Float2 a, const Float2& b) {
    a += b;
    return a;
  }
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct GradOp {
  GradOp(accscalar_t m, const PTA& i, const PTA& g)
    : mean(m), input(i), grad_output(g) {}
  inline Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output[batch][plane][n];
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PTA& input;
  const PTA& grad_output;
};

} // namespace at::native
