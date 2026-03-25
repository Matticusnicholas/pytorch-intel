// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/Distributions.cpp

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/opencl/Distributions.h>
#include <ATen/TensorIterator.h>
#include <ATen/opencl/OpenCLGeneratorImpl.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_dirichlet_grad_native.h>
#include <ATen/ops/_sample_dirichlet_native.h>
#include <ATen/ops/_standard_gamma_grad_native.h>
#include <ATen/ops/_standard_gamma_native.h>
#include <ATen/ops/binomial_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/poisson_native.h>
#endif

namespace at::native {

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor _s_poisson_opencl(const Tensor& lambda, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<OpenCLGeneratorImpl>(gen_, opencl::detail::getDefaultOpenCLGenerator());
  Tensor ret = at::empty(lambda.sizes(), lambda.options());
  launch_poisson_opencl_kernel(ret, lambda, gen);
  return ret;
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor _s_binomial_opencl(const Tensor& count, const Tensor& prob, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<OpenCLGeneratorImpl>(gen_, opencl::detail::getDefaultOpenCLGenerator());
  Tensor ret = at::empty(count.sizes(), count.options());
  at::TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(count)
      .add_input(prob)
      .build();
  launch_binomial_opencl_kernel(iter, gen);
  return ret;
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor _s_gamma_opencl(const Tensor& alpha, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<OpenCLGeneratorImpl>(gen_, opencl::detail::getDefaultOpenCLGenerator());
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  launch_gamma_kernel(ret, alpha, gen);
  return ret;
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor _s_dirichlet_opencl(const Tensor& alpha, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<OpenCLGeneratorImpl>(gen_, opencl::detail::getDefaultOpenCLGenerator());
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  launch_gamma_kernel(ret, alpha, gen);
  auto gamma_sum = ret.sum(/*dim=*/-1, /*keepdim=*/true);
  at::TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(ret)
      .add_input(gamma_sum)
      .build();
  launch_dirichlet_kernel(iter);
  return ret;
}

Tensor _standard_gamma_grad_opencl(const Tensor& self, const Tensor& output) {
  Tensor ret = at::empty(self.sizes(), self.options());
  TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(self)
      .add_input(output)
      .build();
  launch_standard_gamma_grad_kernel(iter);
  return ret;
}

Tensor _dirichlet_grad_opencl(const Tensor& x, const Tensor& alpha, const Tensor& total) {
  Tensor ret = at::empty(x.sizes(), x.options());
  TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(x)
      .add_input(alpha)
      .add_input(total)
      .build();
  launch_dirichlet_grad_kernel(iter);
  return ret;
}

} // namespace at::native
