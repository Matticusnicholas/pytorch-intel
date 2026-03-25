// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ActivationHardtanhKernel.cu
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

void hardtanh_backward_kernel(
    TensorIterator& iter,
    const Scalar& min,
    const Scalar& max) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.dtype(), "hardtanh_backward_opencl", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto min_val = min.to<opmath_t>();
        auto max_val = max.to<opmath_t>();
        sycl_kernel(
            iter,
            [min_val, max_val] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
              opmath_t aop = static_cast<opmath_t>(a);
              opmath_t bop = static_cast<opmath_t>(b);
              return (bop <= min_val) || (bop >= max_val) ? opmath_t(0) : aop;
            });
      });
}
} // namespace

REGISTER_DISPATCH(hardtanh_backward_stub, &hardtanh_backward_kernel)

} // namespace at::native
