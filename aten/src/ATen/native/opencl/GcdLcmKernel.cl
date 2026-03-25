// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/GcdLcmKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
// SYCL: JIT compilation not used, using direct kernel dispatch
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/sycl/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
// SYCL: JIT utilities not used

// NOTE: CUDA on Windows requires that the enclosing function
// of a lambda not have internal linkage.

namespace at::native {

// See note [Jiterator]
constexpr char gcd_name[] = "gcd";
void gcd_kernel_cuda(TensorIteratorBase& iter) {
  // SYCL: Using direct kernel dispatch (no JIT)
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "gcd_opencl", [&]() {
      sycl_kernel(iter, [] SYCL_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        return calc_gcd(a, b);
      });
    });
}

// See note [Jiterator]
constexpr char lcm_name[] = "lcm";
void lcm_kernel_cuda(TensorIteratorBase& iter) {
  // SYCL: Using direct kernel dispatch (no JIT)
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "lcm_opencl", [&]() {
      sycl_kernel(iter, [] SYCL_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        scalar_t g = calc_gcd(a, b);
        return (g == 0) ? 0 : ::abs(a / g * b);
      });
    });
}

REGISTER_DISPATCH(gcd_stub, &gcd_kernel_opencl)
REGISTER_DISPATCH(lcm_stub, &lcm_kernel_opencl)

} // namespace at::native
