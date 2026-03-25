// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ReduceLogicKernel.cu

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/sycl/Reduce.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/Dispatch.h>

namespace at::native {

void and_kernel_opencl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "and_opencl", [&]() {
        sycl_reduce_kernel<scalar_t, bool>(
            iter,
            func_wrapper<bool>([] SYCL_LAMBDA(bool acc, scalar_t val) -> bool {
              return (acc && static_cast<bool>(val));
            }),
            true);
      });
}

void or_kernel_opencl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "or_opencl", [&]() {
        sycl_reduce_kernel<scalar_t, bool>(
            iter,
            func_wrapper<bool>([] SYCL_LAMBDA(bool acc, scalar_t val) -> bool {
              return (acc || static_cast<bool>(val));
            }),
            false);
      });
}

REGISTER_DISPATCH(and_stub, &and_kernel_opencl)
REGISTER_DISPATCH(or_stub, &or_kernel_opencl)

} // namespace at::native
