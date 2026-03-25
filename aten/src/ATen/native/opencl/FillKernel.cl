// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/FillKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Fill.h>
#include <c10/core/Scalar.h>

namespace at::native {

template<typename scalar_t>
struct FillFunctor {
  FillFunctor(scalar_t v): value(v) {}
  inline scalar_t operator() () const {
    return value;
  }
  private:
    scalar_t value;
};

void fill_kernel_opencl(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_V2(iter.dtype(), "fill_opencl", AT_WRAP([&]() {
    sycl_kernel(iter, FillFunctor<scalar_t>(value.to<scalar_t>()));
  }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kBool, kHalf, kBFloat16, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

REGISTER_DISPATCH(fill_stub, &fill_kernel_opencl)

} // namespace at::native
