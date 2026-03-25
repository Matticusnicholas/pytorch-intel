// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ActivationThresholdKernel.cu
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

template <typename scalar_t>
void threshold_kernel_impl(
    TensorIteratorBase& iter,
    scalar_t threshold,
    scalar_t value) {
  sycl_kernel_with_scalars(
      iter, [=] SYCL_LAMBDA(scalar_t x, scalar_t other) -> scalar_t {
        return x <= threshold ? value : other;
      });
}

static void threshold_kernel_opencl(
    TensorIteratorBase& iter,
    const Scalar& threshold,
    const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "threshold_opencl",
      [&] {
        threshold_kernel_impl<scalar_t>(
            iter, threshold.to<scalar_t>(), value.to<scalar_t>());
      });
}

} // namespace

REGISTER_DISPATCH(threshold_stub, &threshold_kernel_opencl)

} // namespace at::native
