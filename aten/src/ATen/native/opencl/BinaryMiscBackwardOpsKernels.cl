// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/BinaryMiscBackwardOpsKernels.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/BinaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/sycl/Loops.h>
// SYCL: JIT compilation not used, using direct kernel dispatch

namespace at::native {

void sigmoid_backward_kernel_opencl(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if(isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "sigmoid_backward_opencl", [&]() {
      sycl_kernel(iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        using comp_t = at::opmath_type<scalar_t>;
        const auto one = comp_t{1.};
        const auto comp_b = static_cast<comp_t>(b);
        const auto comp_a = static_cast<comp_t>(a);
        return static_cast<scalar_t>(comp_a * std::conj((one - comp_b) * comp_b));
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, dtype, "sigmoid_backward_opencl", [&]() {
      sycl_kernel(iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a * (scalar_t(1.) - b) * b;
      });
    });
  }
}

void logit_backward_kernel_opencl(TensorIteratorBase& iter, const Scalar& eps_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.dtype(), "logit_opencl",
      [&]() {
        using T_ACC = acc_type<scalar_t, true>;
        const T_ACC eps = eps_scalar.to<T_ACC>();
        if (eps < T_ACC(0)) {
          sycl_kernel(
              iter, [] SYCL_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
                const T_ACC dy_acc = static_cast<T_ACC>(dy);
                const T_ACC x_acc = static_cast<T_ACC>(x);
                return (x_acc < T_ACC(0) || x_acc > T_ACC(1))
                    ? std::numeric_limits<T_ACC>::quiet_NaN()
                    : dy_acc / (x_acc * (T_ACC(1) - x_acc));
              });
        } else {
          const T_ACC lo = eps;
          const T_ACC hi = T_ACC(1) - eps;
          sycl_kernel(
              iter, [lo, hi] SYCL_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
                const T_ACC dy_acc = static_cast<T_ACC>(dy);
                const T_ACC x_acc = static_cast<T_ACC>(x);
                return (x_acc < lo || x_acc > hi)
                    ? T_ACC(0)
                    : dy_acc / (x_acc * (T_ACC(1) - x_acc));
              });
        }
      });
}

void tanh_backward_kernel_opencl(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if(isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "tanh_backward_complex_opencl", [&]() {
      sycl_kernel(iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        using comp_t = at::opmath_type<scalar_t>;
        const auto one = comp_t{1.};
        const auto comp_b = static_cast<comp_t>(b);
        const auto comp_a = static_cast<comp_t>(a);
        return static_cast<scalar_t>(comp_a * std::conj(one - comp_b * comp_b));
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, dtype, "tanh_backward_opencl", [&]() {
      sycl_kernel(iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a * (scalar_t{1.} - b * b);
      });
    });
  }
}

REGISTER_DISPATCH(sigmoid_backward_stub, &sigmoid_backward_kernel_opencl)
REGISTER_DISPATCH(logit_backward_stub, &logit_backward_kernel_opencl)
REGISTER_DISPATCH(tanh_backward_stub, &tanh_backward_kernel_opencl)

} // namespace at::native
