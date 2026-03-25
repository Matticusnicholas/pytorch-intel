// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/BinaryLogicalOpsKernels.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/sycl/Loops.h>
// SYCL: JIT compilation not used, using direct kernel dispatch
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

namespace at::native {

void logical_and_kernel_opencl(TensorIterator& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_and_opencl", [&]() {
      opmath_symmetric_sycl_kernel_with_scalars<scalar_t, bool>(
          iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a && b;
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, ScalarType::BFloat16,
                               dtype, "logical_and_opencl", [&]() {
      opmath_symmetric_sycl_kernel_with_scalars<scalar_t, bool>(
          iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a && b;
      });
   });
  }
}

void logical_or_kernel_opencl(TensorIterator& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_or_opencl", [&]() {
      sycl_kernel_with_scalars(iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a || b;
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, ScalarType::BFloat16,
                               dtype, "logical_or_opencl", [&]() {
      opmath_symmetric_sycl_kernel_with_scalars<scalar_t, bool>(
          iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return a || b;
      });
    });
  }
}

void logical_xor_kernel_opencl(TensorIterator& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_xor_opencl", [&]() {
      sycl_kernel_with_scalars(iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return bool(a) != bool(b);
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, ScalarType::BFloat16,
                               dtype, "logical_xor_opencl", [&]() {
      opmath_symmetric_sycl_kernel_with_scalars<scalar_t, bool>(
          iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> bool {
        return bool(a) != bool(b);
      });
    });
  }
}

REGISTER_DISPATCH(logical_and_stub, &logical_and_kernel_opencl)
REGISTER_DISPATCH(logical_or_stub, &logical_or_kernel_opencl)
REGISTER_DISPATCH(logical_xor_stub, &logical_xor_kernel_opencl)

} // namespace at::native
