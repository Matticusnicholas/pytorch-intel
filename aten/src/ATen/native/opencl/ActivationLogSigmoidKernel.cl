// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ActivationLogSigmoidKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

// SYCL: thrust not available, using oneDPL or manual implementation

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/sycl/SYCLMathCompat.h>
#include <ATen/native/sycl/ApplyGridUtils.h>
#include <ATen/native/sycl/OffsetCalculator.h>
#include <ATen/native/sycl/Loops.h>

namespace at::native {

// -----------------------------------
// log_sigmoid forward
// -----------------------------------

void launch_log_sigmoid_forward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "log_sigmoid_forward_opencl", [&] {
        using opmath_t = at::opmath_type<scalar_t>;

        sycl_kernel(iter, [] SYCL_LAMBDA(scalar_t in_) -> scalar_t {
          const opmath_t in = in_;
          const auto min = std::min(opmath_t(0), in);
          const auto z = std::exp(-std::abs(in));
          return min - std::log1p(z);
        });
      });
}

namespace {
// -----------------------------------
// log_sigmoid backward
// -----------------------------------
void log_sigmoid_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "log_sigmoid_backward_opencl", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        sycl_kernel(
            iter, [] SYCL_LAMBDA(scalar_t in_, scalar_t grad_out_) -> scalar_t {
              const opmath_t in = in_;
              const opmath_t grad_out = grad_out_;

              auto in_negative = in < opmath_t(0);
              auto max_deriv = in_negative ? opmath_t(1) : opmath_t(0);
              auto sign = in_negative ? opmath_t(1) : -opmath_t(1);
              const auto z = std::exp(-std::abs(in));
              return grad_out * (max_deriv - sign * (z / (opmath_t(1) + z)));
            });
      });
}
} // namespace

REGISTER_DISPATCH(log_sigmoid_backward_stub, &log_sigmoid_backward_kernel)

} // namespace at::native
