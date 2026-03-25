// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/BinaryDivFloorKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/sycl/BinaryInternal.h>
#include <c10/sycl/SYCLGuard.h>
#include <c10/sycl/SYCLMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/generic_math.h>
// SYCL: JIT compilation not used, using direct kernel dispatch
#include <ATen/native/sycl/Loops.h>

#include <type_traits>

namespace at::native {
namespace binary_internal {

void div_floor_kernel_opencl(TensorIteratorBase& iter) {
  const auto dtype = iter.common_dtype();
  if (dtype == kByte) {
    return div_trunc_kernel_opencl(iter);
  } else if (isIntegralType(dtype, /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_floor_opencl", [&]() {
      sycl_kernel_with_scalars(
          iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
            return c10::div_floor_integer(a, b);
      });
    });
  } else if (iter.is_cpu_scalar(2)) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_floor_opencl", [&]() {
          using accscalar_t = at::acc_type<scalar_t, true>;
          auto b = iter.scalar_value<accscalar_t>(2);
          if (C10_UNLIKELY(b == 0)) {
            return div_true_kernel_opencl(iter);
          }
          auto inv_b = accscalar_t(1.0) / b;
          iter.remove_operand(2);
          sycl_kernel(iter, [b, inv_b] SYCL_LAMBDA(scalar_t a) -> scalar_t {
            auto mod = std::fmod(a, b);
            auto div = (a - mod) * inv_b;
            if ((mod != 0) && (b < 0) != (mod < 0)) {
              div -= scalar_t(1);
            }
            scalar_t floordiv;
            if (div != 0) {
              floordiv = std::floor(div);
              if (div - floordiv > scalar_t(0.5)) {
                floordiv += scalar_t(1.0);
              }
            } else {
              floordiv = sycl::copysign(scalar_t(0), a * inv_b);
            }
            return floordiv;
          });
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_floor_opencl", [&]() {
          sycl_kernel_with_scalars(
              iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
                return c10::div_floor_floating(a, b);
              });
        });
  }
}
} // namespace binary_internal

REGISTER_DISPATCH(div_floor_stub, &binary_internal::div_floor_kernel_opencl)

} // namespace at::native
