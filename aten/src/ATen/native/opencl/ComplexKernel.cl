// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ComplexKernel.cu
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/sycl/Loops.h>

namespace at::native {
namespace {

void complex_kernel_opencl(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.input_dtype(0), "complex_opencl", [&]() {
    sycl_kernel(
      iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
        return c10::complex<scalar_t>(a, b);
      });
  });
}

void polar_kernel_opencl(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(0), "polar_opencl", [&]() {
    sycl_kernel(
      iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
        return c10::complex<scalar_t>(a * std::cos(b), a * std::sin(b));
      });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(complex_stub, &complex_kernel_opencl)
REGISTER_DISPATCH(polar_stub, &polar_kernel_opencl)

} // namespace at::native
