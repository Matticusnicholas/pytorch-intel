// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ActivationEluKernel.cu
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
namespace {

void elu_kernel(
    TensorIteratorBase& iter,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "elu_opencl",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto negcoef = alpha.to<opmath_t>() * scale.to<opmath_t>();
        auto poscoef = scale.to<opmath_t>();
        auto negiptcoef = input_scale.to<opmath_t>();
        sycl_kernel(
            iter,
            [negcoef, poscoef, negiptcoef] SYCL_LAMBDA(scalar_t a) -> scalar_t {
              opmath_t aop = static_cast<opmath_t>(a);
              return aop > 0 ? aop * poscoef
                             : std::expm1(aop * negiptcoef) * negcoef;
            });
      });
}

void elu_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "elu_backward_opencl",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto negcoef = alpha.to<opmath_t>() * scale.to<opmath_t>();
        auto poscoef = scale.to<opmath_t>();
        auto negiptcoef = input_scale.to<opmath_t>();
        sycl_kernel(
            iter,
            [negcoef, poscoef, negiptcoef, is_result] SYCL_LAMBDA(
                scalar_t a, scalar_t b) -> scalar_t {
              opmath_t aop = static_cast<opmath_t>(a);
              opmath_t bop = static_cast<opmath_t>(b);

              if (is_result) {
                return bop <= 0 ? aop * negiptcoef * (bop + negcoef)
                                : aop * poscoef;
              } else {
                return bop <= 0
                    ? aop * negiptcoef * negcoef * std::exp(bop * negiptcoef)
                    : aop * poscoef;
              }
            });
      });
}
} // namespace

REGISTER_DISPATCH(elu_stub, &elu_kernel)
REGISTER_DISPATCH(elu_backward_stub, &elu_backward_kernel)

} // namespace at::native
