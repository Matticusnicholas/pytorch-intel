// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/PointwiseOpsKernel.cu

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/native/sycl/Loops.h>
// SYCL: JIT compilation not used, using direct kernel dispatch
#include <ATen/native/sycl/DeviceAddCmulCdiv.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/PointwiseOps.h>
#include <c10/core/Scalar.h>

namespace at::native {

void addcmul_cuda_scalar_tensor2_kernel(
  TensorIteratorBase& iter,
  const Scalar& scalar_tensor2,
  const Scalar& value
);

// SYCL: JIT path disabled, using direct dispatch
#if 0 /* AT_USE_JITERATOR - disabled for SYCL */
constexpr char addcmul_name[] = "addcmul";
#endif
void addcmul_opencl_kernel(TensorIteratorBase& iter, const Scalar& value) {
  TORCH_CHECK(
    !iter.is_cpu_scalar(1),
    "CPU Scalar support for self argument is not supported when "
    "calling addcmul on CUDA tensors."
  );

  TORCH_CHECK(
    !iter.is_cpu_scalar(2),
    "CPU Scalar support for tensor1 argument is not supported when "
    "calling addcmul on CUDA tensors. "
    "However, CPU Scalar support for tensor2 is supported, "
    "please swap your tensor1 and tensor2 terms."
  );

  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    // SYCL: JIT path disabled, using direct dispatch
#if 0 /* AT_USE_JITERATOR - disabled for SYCL */
      AT_DISPATCH_COMPLEX_TYPES(dtype, "addcmul_opencl", [&]() {
        auto alpha = value.to<scalar_t>();
        static const auto addcmul_string = jiterator_stringify(
          template <typename T> T addcmul(T a, T b, T c, T alpha) { return a + alpha * (b * c); });
        if (iter.is_cpu_scalar(3)) {
          auto tensor2_val = iter.scalar_value<scalar_t>(3);
          iter.remove_operand(3);
          return addcmul_cuda_scalar_tensor2_kernel(iter, tensor2_val, value);
        }
        jitted_gpu_kernel<
            /*name=*/addcmul_name,
            /*return_dtype=*/scalar_t,
            /*common_dtype=*/scalar_t,
            /*arity=*/3>(
            iter,
            addcmul_string,
            /*scalar_pos=*/at::cuda::jit::BinaryFuncVariant::NoScalar,
            /*scalar_val=*/0,
            /*extra_args=*/std::make_tuple(alpha));
      });
    #else
      AT_DISPATCH_COMPLEX_TYPES(dtype, "addcmul_opencl", [&]() {
        if (iter.is_cpu_scalar(3)) {
          auto tensor2_val = iter.scalar_value<scalar_t>(3);
          iter.remove_operand(3);
          return addcmul_cuda_scalar_tensor2_kernel(iter, tensor2_val, value);
        }

        auto alpha = value.to<scalar_t>();
        sycl_kernel(iter, [alpha]SYCL_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
          return a + alpha * b * c;
        });
      });
    #endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, dtype, "addcmul_opencl", [&]() {
      if (iter.is_cpu_scalar(3)) {
          auto tensor2_val = iter.scalar_value<scalar_t>(3);
          iter.remove_operand(3);
          return addcmul_cuda_scalar_tensor2_kernel(iter, tensor2_val, value);
      }
      // note(mkozuki): If scalar_t is fp16 or bfloat16, cast scalar to float
      // and do math in fp32 for better accuracy.
      using accscalar_t = at::acc_type<scalar_t, /*is_sycl*/true>;
      auto alpha = value.to<accscalar_t>();
      sycl_kernel(iter, [alpha]SYCL_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
        return pointwise_op_impl<accscalar_t>(a, b, c, alpha, std::multiplies<accscalar_t>());
      });
    });
  }
}

// SYCL: JIT path disabled, using direct dispatch
#if 0 /* AT_USE_JITERATOR - disabled for SYCL */
constexpr char addcmul_scalar_tensor2_name[] = "addcmul_scalar_tensor2";
#endif
void addcmul_cuda_scalar_tensor2_kernel(TensorIteratorBase& iter, const Scalar& scalar_tensor2, const Scalar& value) {
  auto dtype = iter.common_dtype();

  if (at::isComplexType(dtype)) {
    // SYCL: JIT path disabled, using direct dispatch
#if 0 /* AT_USE_JITERATOR - disabled for SYCL */
      AT_DISPATCH_COMPLEX_TYPES(dtype, "addcmul_opencl", [&]() {
        auto c = scalar_tensor2.to<scalar_t>();
        auto alpha = value.to<scalar_t>();

        static const auto addcmul_scalar_tensor2_string = jiterator_stringify(
          template <typename T> T addcmul_scalar_tensor2(T a, T b, T c, T alpha) { return a + alpha * (b * c); });

        jitted_gpu_kernel<
            /*name=*/addcmul_scalar_tensor2_name,
            /*return_dtype=*/scalar_t,
            /*common_dtype=*/scalar_t,
            /*arity=*/2>(
            iter,
            addcmul_scalar_tensor2_string,
            /*scalar_pos=*/at::cuda::jit::BinaryFuncVariant::NoScalar,
            /*scalar_val=*/0,
            /*extra_args=*/std::make_tuple(c, alpha));
        });
    #else
      AT_DISPATCH_COMPLEX_TYPES(dtype, "addcmul_opencl", [&]() {
        auto c = scalar_tensor2.to<scalar_t>();
        auto alpha = value.to<scalar_t>();
        sycl_kernel(iter, [alpha, c]SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return a + alpha * (b * c);
        });
      });
    #endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, dtype, "addcmul_opencl", [&]() {
      // note(mkozuki): If scalar_t is fp16 or bfloat16, cast scalar to float
      // and do math in fp32 for better accuracy.
      using accscalar_t = at::acc_type<scalar_t, /*is_sycl*/true>;
      auto c = scalar_tensor2.to<accscalar_t>();
      auto alpha = value.to<accscalar_t>();
      sycl_kernel(iter, [alpha, c]SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return pointwise_op_impl<accscalar_t>(a, b, c, alpha, std::multiplies<accscalar_t>());
      });
    });
  }
}

// SYCL: JIT path disabled, using direct dispatch
#if 0 /* AT_USE_JITERATOR - disabled for SYCL */
// return a + alpha * (b / static_cast<accscalar_t>(c));
constexpr char addcdiv_name[] = "addcdiv";
#endif
void addcdiv_opencl_kernel(TensorIteratorBase& iter, const Scalar& value) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    // SYCL: JIT path disabled, using direct dispatch
#if 0 /* AT_USE_JITERATOR - disabled for SYCL */
      AT_DISPATCH_COMPLEX_TYPES(dtype, "addcdiv_opencl", [&]() {
        auto alpha = value.to<scalar_t>();
        static const auto addcdiv_string =
            jiterator_stringify(template <typename T> T addcdiv(
                T a, T b, T c, T alpha) { return a + alpha * (b / c); });
        jitted_gpu_kernel<
            /*name=*/addcdiv_name,
            /*return_dtype=*/scalar_t,
            /*common_dtype=*/scalar_t,
            /*arity=*/3>(
            iter,
            addcdiv_string,
            /*scalar_pos=*/at::cuda::jit::BinaryFuncVariant::NoScalar,
            /*scalar_val=*/0,
            /*extra_args=*/std::make_tuple(alpha));
      });
    #else
      AT_DISPATCH_COMPLEX_TYPES(dtype, "addcdiv_opencl", [&]() {
        auto alpha = value.to<scalar_t>();
        sycl_kernel(iter, [alpha]SYCL_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
          return a + alpha * (b / c);
        });
      });
    #endif
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, dtype, "addcdiv_opencl", [&]() {
      // note(mkozuki): If scalar_t is fp16 or bfloat16, cast scalar to float
      // and do math in fp32 for better accuracy.
      using accscalar_t = at::acc_type<scalar_t, /*is_sycl*/true>;
      auto alpha = value.to<accscalar_t>();
      sycl_kernel(iter, [alpha]SYCL_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
        //return a + alpha * (b / static_cast<accscalar_t>(c));
        return pointwise_op_impl<accscalar_t>(a, b, c, alpha, std::divides<accscalar_t>());
      });
    });
  }
}

void smooth_l1_backward_opencl_kernel(TensorIterator& iter, const Scalar& norm, double beta) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "smooth_l1_backward_opencl", [&iter, &norm, beta] {
      auto norm_val = norm.to<scalar_t>();
      scalar_t beta_val(beta);
      sycl_kernel(iter, [norm_val, beta_val]SYCL_LAMBDA(scalar_t input, scalar_t target, scalar_t grad_output) -> scalar_t {
        const auto x = input - target;
        if (x < -beta_val)
          return -norm_val * grad_output;
        else if (x > beta_val)
          return norm_val * grad_output;
        else
          return norm_val * x * grad_output / beta_val;
    });
  });
}

void huber_backward_opencl_kernel(TensorIterator& iter, const Scalar& norm, double delta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "huber_backward_opencl", [&iter, &norm, delta] {
    auto norm_val = norm.to<scalar_t>();
    scalar_t delta_val(delta);
    sycl_kernel(iter, [norm_val, delta_val]SYCL_LAMBDA(scalar_t input, scalar_t target, scalar_t grad_output) -> scalar_t {
      const auto x = input - target;
      if (x < -delta_val) {
        return -norm_val * grad_output * delta_val;
      } else if (x > delta_val) {
        return norm_val * grad_output * delta_val;
      } else {
        return norm_val * x * grad_output;
      }
    });
  });
}

void mse_backward_opencl_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "mse_backward_opencl", [&]() {
    auto alpha = value.to<scalar_t>();
    sycl_kernel(iter, [alpha]SYCL_LAMBDA(scalar_t a, scalar_t b, scalar_t c) -> scalar_t {
      return alpha * (a - b) * c;
    });
  });
}

REGISTER_DISPATCH(addcdiv_stub, &addcdiv_opencl_kernel)
REGISTER_DISPATCH(addcmul_stub, &addcmul_opencl_kernel)
REGISTER_DISPATCH(smooth_l1_backward_stub, &smooth_l1_backward_opencl_kernel)
REGISTER_DISPATCH(huber_backward_stub, &huber_backward_opencl_kernel)
REGISTER_DISPATCH(mse_backward_stub, &mse_backward_opencl_kernel)
} // namespace at::native
