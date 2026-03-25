// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ActivationPreluKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

// SYCL: thrust not available, using oneDPL or manual implementation
#include <tuple>

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
// prelu
// -----------------------------------
void prelu_kernel(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "prelu_opencl", [&] {
    sycl_kernel(iter,
      [] SYCL_LAMBDA (scalar_t input, scalar_t weight) -> scalar_t {
        return (input > 0) ? input : weight * input;
      });
  });
}

void prelu_backward_kernel(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "prelu_backward_opencl", [&] {
    sycl_kernel_multiple_outputs(iter,
      [] SYCL_LAMBDA (scalar_t input, scalar_t weight, scalar_t grad) -> std::tuple<scalar_t, scalar_t> {
        auto mask = input > 0;
        auto grad_input = mask ? grad : weight * grad;
        auto grad_weight = mask ? scalar_t{0} : input * grad;
        return {grad_input, grad_weight};
      });
  });
}

REGISTER_DISPATCH(prelu_stub, &prelu_kernel)
REGISTER_DISPATCH(prelu_backward_stub, &prelu_backward_kernel)

} // namespace at::native
