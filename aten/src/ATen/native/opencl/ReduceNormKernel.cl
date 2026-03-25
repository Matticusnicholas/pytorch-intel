// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ReduceNormKernel.cu

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/sycl/Reduce.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/LinearAlgebra.h>
#include <c10/core/Scalar.h>

namespace at::native {

// This reduction accumulates results as the type `acc_t`. By default, when
// `scalar_t` is complex, `acc_t` is the downgraded real number type.
// Otherwise, `acc_t` and `scalar_t` are the same type.
template <typename scalar_t, typename acc_t=typename scalar_value_type<scalar_t>::type, typename out_t=typename scalar_value_type<scalar_t>::type>
void norm_kernel_opencl_impl(TensorIterator& iter, double p) {
  if (p == static_cast<double>(0)) {
    sycl_reduce_kernel<scalar_t, out_t>(iter, NormZeroOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(1)) {
    sycl_reduce_kernel<scalar_t, out_t>(iter, NormOneOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(2)) {
    sycl_reduce_kernel<scalar_t, out_t>(iter, NormTwoOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(INFINITY)) {
    sycl_reduce_kernel<scalar_t, out_t>(iter, AbsMaxOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(-INFINITY)) {
    sycl_reduce_kernel<scalar_t, out_t>(iter, AbsMinOps<scalar_t, acc_t, out_t>(), std::numeric_limits<acc_t>::infinity());
  } else {
    sycl_reduce_kernel<scalar_t, out_t>(iter, NormOps<scalar_t, acc_t, out_t>{acc_t(p)}, 0);
  }
}

void norm_launch_kernel(TensorIterator& iter, double ord) {
  if (iter.dtype(0) == kHalf) {
    return norm_kernel_opencl_impl<at::Half, float>(iter, ord);
  } else if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_opencl_impl<at::Half, float, float>(iter, ord);
  }
  else if(iter.dtype(0) == kBFloat16) {
    return norm_kernel_opencl_impl<at::BFloat16, float>(iter, ord);
  } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_opencl_impl<at::BFloat16, float, float>(iter, ord);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.input_dtype(), "norm_opencl", [&] {
    norm_kernel_opencl_impl<scalar_t>(iter, ord);
  });
}

// powsum: computes sum(|x|^p) without the final root
template <typename scalar_t, typename acc_t=typename scalar_value_type<scalar_t>::type, typename out_t=typename scalar_value_type<scalar_t>::type>
void powsum_kernel_opencl_impl(TensorIterator& iter, double p) {
  if (p == static_cast<double>(2)) {
    sycl_reduce_kernel<scalar_t, out_t>(iter, NormTwoOps<scalar_t, acc_t, out_t, false>(), 0);
  } else {
    sycl_reduce_kernel<scalar_t, out_t>(iter, NormOps<scalar_t, acc_t, out_t, false>{acc_t(p)}, 0);
  }
}

void powsum_launch_kernel(TensorIterator& iter, double ord) {
  if (iter.dtype(0) == kHalf) {
    return powsum_kernel_opencl_impl<at::Half, float>(iter, ord);
  } else if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
    return powsum_kernel_opencl_impl<at::Half, float, float>(iter, ord);
  }
  else if(iter.dtype(0) == kBFloat16) {
    return powsum_kernel_opencl_impl<at::BFloat16, float>(iter, ord);
  } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
    return powsum_kernel_opencl_impl<at::BFloat16, float, float>(iter, ord);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.input_dtype(), "powsum_opencl", [&] {
    powsum_kernel_opencl_impl<scalar_t>(iter, ord);
  });
}

} // namespace at::native
