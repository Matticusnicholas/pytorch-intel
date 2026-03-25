// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/BinaryMulKernel.cu
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
// SYCL: JIT compilation not used, using direct kernel dispatch
#include <ATen/native/sycl/Loops.h>

#include <type_traits>

namespace at::native {

void mul_kernel_opencl(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
    // SYCL: JIT not supported, using direct dispatch
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_symmetric_sycl_kernel_with_scalars<scalar_t>(
        iter, binary_internal::MulFunctor<opmath_t>());
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "mul_opencl", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          opmath_symmetric_sycl_kernel_with_scalars<scalar_t>(
              iter, binary_internal::MulFunctor<opmath_t>());
        });
  }
}

REGISTER_DISPATCH(mul_stub, &mul_kernel_opencl)

} // namespace at::native
