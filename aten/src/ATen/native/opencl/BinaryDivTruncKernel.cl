// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/BinaryDivTruncKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <c10/sycl/SYCLGuard.h>
#include <c10/sycl/SYCLMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>
// SYCL: JIT compilation not used, using direct kernel dispatch
#include <ATen/native/sycl/Loops.h>

#include <type_traits>

namespace at::native {
namespace binary_internal {

void div_trunc_kernel_opencl(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (isIntegralType(dtype, /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_trunc_opencl", [&]() {
      sycl_kernel_with_scalars(
          iter,
          [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t { return a / b; });
    });
  } else if (iter.is_cpu_scalar(2)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_trunc_opencl", [&]() {
          using accscalar_t = at::acc_type<scalar_t, true>;
          auto inv_b = accscalar_t(1.0) / iter.scalar_value<accscalar_t>(2);
          iter.remove_operand(2);
          sycl_kernel(iter, [inv_b] SYCL_LAMBDA(scalar_t a) -> scalar_t {
            return std::trunc(a * inv_b);
          });
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_trunc_opencl", [&]() {
          sycl_kernel_with_scalars(
              iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
                return std::trunc(a / b);
              });
        });
  }
}
} // namespace binary_internal

REGISTER_DISPATCH(div_trunc_stub, &binary_internal::div_trunc_kernel_opencl)

} // namespace at::native
