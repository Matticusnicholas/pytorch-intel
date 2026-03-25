// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/AbsKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>
#include <ATen/native/sycl/Loops.h>
// SYCL: JIT compilation not used, using direct kernel dispatch
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

namespace at::native {

template<typename scalar_t>
struct AbsFunctor {
  inline scalar_t operator() (const scalar_t a) const {
    return std::abs(a);
  }
};

void abs_kernel_opencl(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "abs_opencl", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      sycl_kernel(iter, AbsFunctor<opmath_t>());
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(
        ScalarType::Half,
        ScalarType::BFloat16,
        ScalarType::Bool,
        iter.dtype(),
        "abs_opencl",
        [&]() { sycl_kernel(iter, AbsFunctor<scalar_t>()); });
  }
}

  REGISTER_DISPATCH(abs_stub, &abs_kernel_opencl)

} // namespace at::native
