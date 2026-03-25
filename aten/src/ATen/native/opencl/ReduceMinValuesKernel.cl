// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ReduceMinValuesKernel.cu

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/sycl/Reduce.h>
#include <ATen/native/cuda/ReduceOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/sycl/NumericLimits.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/NumericUtils.h>

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/sycl/NumericLimits.h>

// SYCL: thrust not available, using oneDPL or manual implementation
// #include <thrust/pair.h>

namespace at::native {

template <typename acc_t>
struct MinNanFunctor {
  inline acc_t operator()(acc_t a, acc_t b) const {
      return (at::_isnan(a) || a < b) ? a : b;
  }
};

template <typename scalar_t, typename acc_t=scalar_t>
void min_values_kernel_opencl_impl(TensorIterator& iter) {
  sycl_reduce_kernel<scalar_t, scalar_t>(
    iter, func_wrapper<acc_t> (MinNanFunctor<acc_t>()),
    at::numeric_limits<acc_t>::upper_bound());
}

void min_values_kernel_opencl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "min_values_opencl", [&]() {
    min_values_kernel_opencl_impl<scalar_t>(iter);
  });
}

void min_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "min_opencl", [&]() {
    sycl_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MinOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(at::numeric_limits<scalar_t>::upper_bound(), 0));
  });
}

void min_all_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "min_all_opencl", [&] {
    min_values_kernel_opencl_impl<scalar_t>(iter);
  });
}

REGISTER_DISPATCH(min_values_stub, &min_values_kernel_opencl)

} // namespace at::native
