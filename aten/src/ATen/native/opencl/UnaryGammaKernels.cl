#define TORCH_ASSERT_NO_OPERATORS
#include <limits>
#include <ATen/native/UnaryOps.h>
// SYCL: JIT compilation not used, using direct kernel dispatch
#include <ATen/native/sycl/Loops.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/sycl/Math.h>
#include <ATen/native/Math.h>

namespace at::native {

#if AT_USE_JITERATOR()
constexpr char digamma_name[] = "digamma";
#endif // AT_USE_JITERATOR()
// See note [Jiterator]
void digamma_kernel_opencl(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(), "digamma_opencl", [&]() {
        jitted_gpu_kernel</*name=*/digamma_name,
                          /*return_dtype=*/ scalar_t,
                          /*common_dtype=*/ scalar_t,
                          /*arity=*/ 1>(iter, digamma_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(), "digamma_opencl", [&]() {
        sycl_kernel(iter, []SYCL_LAMBDA(scalar_t a) -> scalar_t {
          return calc_digamma(a);
        });
    });
  #endif // AT_USE_JITERATOR()
}

// See note [Jiterator]
constexpr char trigamma_name[] = "trigamma";
void trigamma_kernel_opencl(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(), "trigamma_opencl", [&]() {
        jitted_gpu_kernel</*name=*/trigamma_name,
                          /*return_dtype=*/ scalar_t,
                          /*common_dtype=*/ scalar_t,
                          /*arity=*/ 1>(iter, trigamma_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(), "trigamma_opencl", [&]() {
        sycl_kernel(iter, []SYCL_LAMBDA(scalar_t a) -> scalar_t {
          return calc_trigamma(a);
        });
    });
  #endif // AT_USE_JITERATOR()
}

constexpr char polygamma_name[] = "polygamma";
void polygamma_kernel_opencl(TensorIteratorBase& iter, int64_t n) {
  if (n == 0) {
    digamma_kernel_opencl(iter);
  } else if (n == 1) {
    trigamma_kernel_opencl(iter);
  } else {
#if AT_USE_JITERATOR()
    // TODO : `unary_jitted_gpu_kernel` for cleaner UX.
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
        iter.common_dtype(), "polygamma_opencl", [&]() {
          jitted_gpu_kernel<
              /*name=*/polygamma_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(
              iter,
              polygamma_string,
              /*scalar_pos=*/at::sycl::jit::BinaryFuncVariant::NoScalar,
              /*scalar_val=*/0,
              /*extra_args=*/std::make_tuple(n));
        });
#else
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
        iter.common_dtype(), "polygamma_opencl", [&]() {
          sycl_kernel(iter, [=] SYCL_LAMBDA(scalar_t a) -> scalar_t {
            return calc_polygamma<scalar_t, /*is_sycl=*/true>(a, static_cast<int>(n));
          });
        });
#endif // AT_USE_JITERATOR()
  }
}

constexpr char lgamma_name[] = "lgamma_kernel";
void lgamma_kernel_opencl(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(), "lgamma_opencl", [&]() {
        jitted_gpu_kernel</*name=*/lgamma_name,
                          /*return_dtype=*/ scalar_t,
                          /*common_dtype=*/ scalar_t,
                          /*arity=*/ 1>(iter, lgamma_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(), "lgamma_opencl", [&]() {
        sycl_kernel(iter, []SYCL_LAMBDA(scalar_t a) -> scalar_t {
          return ::lgamma(a);
        });
    });
  #endif
}

REGISTER_DISPATCH(digamma_stub, &digamma_kernel_opencl)
REGISTER_DISPATCH(polygamma_stub, &polygamma_kernel_opencl)
REGISTER_DISPATCH(lgamma_stub, &lgamma_kernel_opencl)

} // namespace at::native
