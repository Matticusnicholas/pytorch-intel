// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <c10/sycl/SYCLGuard.h>
#include <c10/sycl/SYCLMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>
#include <ATen/native/sycl/BinaryInternal.h>
// SYCL: JIT compilation not used, using direct kernel dispatch
#include <ATen/native/sycl/Loops.h>

#include <type_traits>

namespace at::native {
namespace binary_internal {

void div_true_kernel_opencl(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (iter.common_dtype() == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
    // SYCL: JIT not supported, using direct dispatch
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_gpu_kernel_with_scalars<scalar_t>(iter, DivFunctor<opmath_t>());
    return;
  }
  if (iter.is_cpu_scalar(2)) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, common_dtype, "div_true_opencl", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          using high_prec_t = std::conditional_t<
              c10::is_complex<scalar_t>::value,
              c10::complex<double>,
              double>;
          auto inv_b = static_cast<opmath_t>(high_prec_t(1.0) / iter.scalar_value<high_prec_t>(2));
          iter.remove_operand(2);
          sycl_kernel(
              iter,
              BUnaryFunctor<scalar_t, scalar_t, scalar_t, MulFunctor<opmath_t>>(
                  MulFunctor<opmath_t>(), inv_b));
        });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, common_dtype, "div_true_opencl", [&]() {
          DivFunctor<scalar_t> f;
          sycl_kernel_with_scalars(iter, f);
        });
  }
}
} // namespace binary_internal

REGISTER_DISPATCH(div_true_stub, &binary_internal::div_true_kernel_opencl)

} // namespace at::native
