// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/DeviceAddCmulCdiv.cuh
#pragma once

#include <cmath>
#include <type_traits>

namespace at::native {

// Computes input + alpha * op(tensor1, tensor2).
// Special-cases alpha=1 and uses explicit std::fma for multiplies.
template <typename opmath_t, typename Op>
C10_HOST_DEVICE inline opmath_t pointwise_op_impl(
    opmath_t input,
    opmath_t tensor1,
    opmath_t tensor2,
    opmath_t alpha,
    Op op) {
  if (alpha == opmath_t(1)) {
    if constexpr (std::is_same_v<Op, std::multiplies<opmath_t>> &&
                  std::is_floating_point_v<opmath_t>) {
      return std::fma(tensor1, tensor2, input);
    } else {
      return input + op(tensor1, tensor2);
    }
  }
  if constexpr(std::is_floating_point_v<opmath_t>) {
    return std::fma(alpha, op(tensor1, tensor2), input);
  } else {
    return input + alpha * op(tensor1, tensor2);
  }
}

} // namespace at::native
