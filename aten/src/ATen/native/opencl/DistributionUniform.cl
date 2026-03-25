// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/DistributionUniform.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/sycl/SYCLGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/sycl/DistributionTemplates.h>

namespace at::native {

void uniform_kernel(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<SYCLGeneratorImpl>(gen, sycl::detail::getDefaultSYCLGenerator());
  templates::cuda::uniform_kernel(iter, from, to, generator);
}

REGISTER_DISPATCH(uniform_stub, &uniform_kernel)

} // namespace at::native
