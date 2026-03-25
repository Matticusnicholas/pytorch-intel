// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/DistributionLogNormalKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/sycl/SYCLGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/sycl/DistributionTemplates.h>

namespace at::native {

void log_normal_kernel(TensorIteratorBase& iter, double mean, double std, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<SYCLGeneratorImpl>(gen, sycl::detail::getDefaultSYCLGenerator());
  at::native::templates::sycl::log_normal_kernel(iter, mean, std, generator);
}

REGISTER_DISPATCH(log_normal_stub, &log_normal_kernel)

} // namespace at::native
