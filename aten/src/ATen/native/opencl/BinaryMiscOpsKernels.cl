// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/BinaryMiscOpsKernels.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/sycl/Math.h>
#include <ATen/NumericUtils.h>

namespace at::native {

void smooth_l1_kernel_opencl(TensorIteratorBase& iter, double beta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "smooth_l1_opencl", [&iter, beta]() {
    scalar_t beta_val(beta);
    sycl_kernel(iter, [beta_val] SYCL_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      auto z = ::abs(a - b);
      return z < beta_val ? scalar_t(0.5) * z * z / beta_val : z - scalar_t(0.5) * beta_val;
    });
  });
}

void huber_kernel_opencl(TensorIterator& iter, double delta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "huber_opencl", [&iter, delta] {
    scalar_t delta_val(delta);
    sycl_kernel(iter, [delta_val] SYCL_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      auto z = ::abs(a - b);
      return z < delta_val ? scalar_t(0.5) * z * z : delta_val * (z - scalar_t(0.5) * delta_val);
    });
  });
}

void mse_kernel_opencl(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "mse_opencl", [&]() {
    sycl_kernel(iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      auto diff = a - b;
      return diff * diff;
    });
  });
}

void xlogy_kernel_opencl(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "xlogy_opencl", [&]() {
    sycl_kernel_with_scalars(iter, [] SYCL_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
      if (at::_isnan(y)) return NAN;
      if (x == 0) return 0;
      return x * std::log(y);
    });
  });
}

void xlog1py_kernel_opencl(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "xlog1py_opencl", [&]() {
    sycl_kernel_with_scalars(iter, [] SYCL_LAMBDA(scalar_t x, scalar_t y) -> scalar_t {
      if (at::_isnan(y)) return NAN;
      if (x == 0) return 0;
      return x * std::log1p(y);
    });
  });
}

void ldexp_kernel_opencl(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(0), "ldexp_opencl", [&] {
    sycl_kernel(iter, [] SYCL_LAMBDA(scalar_t x, int exp) -> scalar_t {
      return ::ldexp(x, exp);
    });
  });
}

REGISTER_DISPATCH(smooth_l1_stub, &smooth_l1_kernel_opencl)
REGISTER_DISPATCH(huber_stub, &huber_kernel_opencl)
REGISTER_DISPATCH(mse_stub, &mse_kernel_opencl)
REGISTER_DISPATCH(xlogy_stub, &xlogy_kernel_opencl)
REGISTER_DISPATCH(xlog1py_stub, &xlog1py_kernel_opencl)
REGISTER_DISPATCH(ldexp_stub, &ldexp_kernel_opencl)

} // namespace at::native
