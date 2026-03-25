// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/DistributionGeometricKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/sycl/SYCLGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/sycl/DistributionTemplates.h>

namespace at::native {

void geometric_kernel(TensorIteratorBase& iter, double p_, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<SYCLGeneratorImpl>(gen, sycl::detail::getDefaultSYCLGenerator());
  at::native::templates::sycl::geometric_kernel(iter, p_, generator);
}

REGISTER_DISPATCH(geometric_stub, &geometric_kernel)

} // namespace at::native
