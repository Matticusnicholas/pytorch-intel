// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/DistributionExponentialKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/sycl/SYCLGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/sycl/DistributionTemplates.h>

namespace at::native {

void exponential_kernel(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<SYCLGeneratorImpl>(gen, sycl::detail::getDefaultSYCLGenerator());
  at::native::templates::sycl::exponential_kernel(iter, lambda, generator);
}

REGISTER_DISPATCH(exponential_stub, &exponential_kernel)

} // namespace at::native
