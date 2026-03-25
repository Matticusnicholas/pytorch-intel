#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <c10/util/BFloat16-math.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a lambda not have internal linkage.

namespace at::native {

void nextafter_kernel_opencl(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "nextafter_opencl", [&]() {
    sycl_kernel_with_scalars(iter, []SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return std::nextafter(a, b);
    });
  });
}

void heaviside_kernel_opencl(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, iter.dtype(), "heaviside_opencl", [&]() {
    sycl_kernel_with_scalars(iter, []SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a == 0 ? b : static_cast<scalar_t>(a > 0);
    });
  });
}

REGISTER_DISPATCH(nextafter_stub, &nextafter_kernel_opencl)
REGISTER_DISPATCH(heaviside_stub, &heaviside_kernel_opencl)

} // namespace at::native
