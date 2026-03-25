// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/DistributionRandomKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/sycl/SYCLGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/sycl/DistributionTemplates.h>

namespace at::native {

void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range, int64_t base, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<SYCLGeneratorImpl>(gen_, sycl::detail::getDefaultSYCLGenerator());
  at::native::templates::sycl::random_from_to_kernel(iter, range, base, gen);
}

void random_full_64_bits_range_kernel(TensorIteratorBase& iter, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<SYCLGeneratorImpl>(gen_, sycl::detail::getDefaultSYCLGenerator());
  at::native::templates::sycl::random_full_64_bits_range_kernel(iter, gen);
}

void random_kernel(TensorIteratorBase& iter, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<SYCLGeneratorImpl>(gen_, sycl::detail::getDefaultSYCLGenerator());
  at::native::templates::sycl::random_kernel(iter, gen);
}

REGISTER_DISPATCH(random_from_to_stub, &random_from_to_kernel)
REGISTER_DISPATCH(random_stub, &random_kernel)
REGISTER_DISPATCH(random_full_64_bits_range_stub, &random_full_64_bits_range_kernel)

} // namespace at::native
