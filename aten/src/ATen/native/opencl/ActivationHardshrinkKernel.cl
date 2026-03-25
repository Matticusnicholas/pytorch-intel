// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ActivationHardshrinkKernel.cu
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

void hardshrink_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardshrink_opencl",
      [&]() {
        auto lambd = value.to<scalar_t>();
        sycl_kernel(iter, [lambd] SYCL_LAMBDA(scalar_t a) -> scalar_t {
          return (a >= -lambd && a <= lambd) ? scalar_t(0) : a;
        });
      });
}
} // namespace

REGISTER_DISPATCH(hardshrink_stub, &hardshrink_kernel)

} // namespace at::native
