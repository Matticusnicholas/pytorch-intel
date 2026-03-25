// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/CopysignKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

#include <c10/sycl/SYCLMathCompat.h>

namespace at::native {

void copysign_kernel_opencl(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "copysign_opencl", [&]() {
    sycl_kernel_with_scalars(iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return sycl::copysign(a, b);
    });
  });
}

REGISTER_DISPATCH(copysign_stub, &copysign_kernel_opencl)

} // namespace at::native
