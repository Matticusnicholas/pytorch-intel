// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/BinaryRemainderKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/TypeSafeSignMath.h>

#include <limits>
#include <type_traits>

namespace at::native {

void remainder_kernel_opencl(TensorIteratorBase& iter) {
  if (isIntegralType(iter.common_dtype(), /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "remainder_opencl", [&]() {
      sycl_kernel_with_scalars(iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        if constexpr (std::is_same_v<scalar_t, uint8_t>) {
          if (b == 0) {
            return std::numeric_limits<uint8_t>::max();
          }
        }
        scalar_t r = a % b;
        if (r != 0 && c10::signs_differ(r, b)) {
          r += b;
        }
        return r;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "remainder_opencl", [&]() {
      sycl_kernel_with_scalars(iter,
        [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          auto mod = ::fmod(a, b);
          if (mod != 0 && c10::signs_differ(b, mod)) {
            mod += b;
          }
          return mod;
        });
    });
  }
}

void fmod_kernel_opencl(TensorIteratorBase& iter) {
  if (isIntegralType(iter.common_dtype(), /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "fmod_opencl", [&]() {
      sycl_kernel_with_scalars(iter, [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a % b;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "fmod_opencl", [&]() {
      sycl_kernel_with_scalars(iter,
        [] SYCL_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return ::fmod(a, b);
        });
    });
  }
}

REGISTER_DISPATCH(remainder_stub, &remainder_kernel_opencl)
REGISTER_DISPATCH(fmod_stub, &fmod_kernel_opencl)

} // namespace at::native
