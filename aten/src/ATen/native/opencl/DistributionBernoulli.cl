// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/DistributionBernoulli.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/sycl/CUDAApplyUtils.h>
#include <ATen/AccumulateType.h>
#include <ATen/sycl/SYCLGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/sycl/DistributionTemplates.h>

// SYCL: #include <curand.h> replaced with oneMKL/oneDPL random
// SYCL: #include <curand_kernel.h> replaced with oneMKL/oneDPL random
// SYCL: #include <curand_philox4x32_x.h> replaced with oneMKL/oneDPL random
#include <utility>
#include <functional>

#include <ATen/native/Distributions.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/TensorIterator.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <type_traits>

namespace at::native {

void bernoulli_tensor_kernel(const TensorBase &self, const TensorBase &p_, std::optional<Generator> gen_) {
  auto generator = get_generator_or_default<SYCLGeneratorImpl>(gen_, sycl::detail::getDefaultSYCLGenerator());
  at::native::templates::sycl::bernoulli_kernel(self, p_, generator);
}

void bernoulli_scalar_kernel(const TensorBase &self, double p, std::optional<Generator> gen) {
  auto iter = TensorIterator::borrowing_nullary_op(self);
  auto generator = get_generator_or_default<SYCLGeneratorImpl>(gen, sycl::detail::getDefaultSYCLGenerator());
  at::native::templates::sycl::bernoulli_kernel(iter, p, generator);
}

REGISTER_DISPATCH(bernoulli_tensor_stub, &bernoulli_tensor_kernel)
REGISTER_DISPATCH(bernoulli_scalar_stub, &bernoulli_scalar_kernel)

} // namespace at::native
