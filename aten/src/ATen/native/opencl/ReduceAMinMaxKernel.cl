// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ReduceAMinMaxKernel.cu

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

template <typename scalar_t>
void _min_max_values_kernel_opencl_impl(TensorIterator& iter) {
  sycl_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MinMaxOps<scalar_t, scalar_t, int32_t>{},
      std::pair<scalar_t, scalar_t>(
          at::numeric_limits<scalar_t>::upper_bound(),
          at::numeric_limits<scalar_t>::lower_bound()));
}

void aminmax_allreduce_launch_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.input_dtype(), "aminmax_all_opencl", [&] {
        _min_max_values_kernel_opencl_impl<scalar_t>(iter);
      });
}

void aminmax_launch_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.input_dtype(), "aminmax_opencl", [&]() {
        sycl_reduce_kernel<scalar_t, scalar_t>(
            iter,
            MinMaxOps<scalar_t, scalar_t, int32_t>{},
            std::pair<scalar_t, scalar_t>(
                at::numeric_limits<scalar_t>::upper_bound(),
                at::numeric_limits<scalar_t>::lower_bound()));
      });
}

} // namespace at::native
