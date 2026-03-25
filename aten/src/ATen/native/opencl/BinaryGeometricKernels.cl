// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/BinaryGeometricKernels.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

namespace at::native {

void atan2_kernel_opencl(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.common_dtype(), "atan2_opencl",
      [&]() {
        sycl_kernel_with_scalars(iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return ::atan2(a, b);
        });
      });
}

void hypot_kernel_opencl(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.common_dtype(), "hypot_opencl",
      [&]() {
        opmath_symmetric_sycl_kernel_with_scalars<scalar_t>(
            iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return ::hypot(a, b);
        });
      });
}

REGISTER_DISPATCH(atan2_stub, &atan2_kernel_opencl)
REGISTER_DISPATCH(hypot_stub, &hypot_kernel_opencl)

} // namespace at::native
