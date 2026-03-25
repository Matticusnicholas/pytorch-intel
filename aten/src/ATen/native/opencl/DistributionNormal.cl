// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/DistributionNormal.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>
#include <ATen/sycl/SYCLGeneratorImpl.h>
#include <ATen/native/sycl/DistributionTemplates.h>

namespace at::native {

void normal_kernel(const TensorBase &self, double mean, double std, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<SYCLGeneratorImpl>(gen, sycl::detail::getDefaultSYCLGenerator());
  at::native::templates::sycl::normal_kernel(self, mean, std, generator);
}

REGISTER_DISPATCH(normal_stub, &normal_kernel)

} // namespace at::native
