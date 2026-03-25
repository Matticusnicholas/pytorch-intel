// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ZetaKernel.cu

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
// SYCL: JIT compilation not used, using direct kernel dispatch
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/sycl/Math.h>
// SYCL: JIT compilation not used, using direct kernel dispatch

namespace at::native {
namespace {

void zeta_kernel_opencl(TensorIteratorBase& iter) {
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "zeta_opencl", [&]() {
      sycl_kernel_with_scalars(iter, []SYCL_LAMBDA(scalar_t x, scalar_t q) -> scalar_t {
        return zeta<scalar_t, /*is_cuda=*/true>(x, q);
      });
    });
}

}  // namespace (anonymous)

REGISTER_DISPATCH(zeta_stub, &zeta_kernel_opencl)

} // namespace at::native
