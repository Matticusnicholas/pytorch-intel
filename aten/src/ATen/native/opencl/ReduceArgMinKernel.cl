// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ReduceArgMinKernel.cu

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/ReduceOps.h>
#include <ATen/sycl/NumericLimits.h>
#include <ATen/native/sycl/Reduce.h>

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/sycl/NumericLimits.h>

// SYCL: thrust not available, using oneDPL or manual implementation
// #include <thrust/pair.h>

namespace at::native {

template <typename scalar_t, typename acc_t = scalar_t>
void argmin_kernel_opencl_impl(TensorIterator& iter) {
  sycl_reduce_kernel<scalar_t, int64_t>(
      iter,
      ArgMinOps<acc_t>{},
      std::pair<acc_t, int64_t>(
          at::numeric_limits<acc_t>::upper_bound(), 0));
};

void argmin_kernel_opencl(TensorIterator& iter) {
  // For float16 & bfloat16, instead of implementing is_nan and warp_shfl_down,
  // we can convert float16 & bfloat16 to float and do all the operations in
  // float.
  if (iter.dtype(1) == kHalf) {
    argmin_kernel_opencl_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kBFloat16) {
    argmin_kernel_opencl_impl<at::BFloat16, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmin_opencl", [&]() {
      argmin_kernel_opencl_impl<scalar_t>(iter);
    });
  }
}

REGISTER_DISPATCH(argmin_stub, &argmin_kernel_opencl)

} // namespace at::native
