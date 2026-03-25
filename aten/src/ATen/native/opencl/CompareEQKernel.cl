// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/CompareEQKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/sycl/Loops.h>

namespace at::native { namespace {

enum class EqOpType {EQ, NE};

template<typename scalar_t>
struct CompareEqFunctor{
  CompareEqFunctor(EqOpType op): op_(op) {}
  const EqOpType op_;
  inline bool operator() (scalar_t a, scalar_t b) const {
    if (op_ == EqOpType::EQ) {
      return a == b;
    } else {
      return a != b;
    }
  }
 };
}

C10_NOINLINE void compare_eq_ne_kernel(TensorIteratorBase &iter, EqOpType op) {
  AT_DISPATCH_V2(iter.common_dtype(), "compare_eq_ne_opencl", AT_WRAP([&]() {
    opmath_symmetric_sycl_kernel_with_scalars<scalar_t, bool>(
        iter, CompareEqFunctor<scalar_t>(op));
  }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kHalf, kBFloat16, kBool, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kFloat4_e2m1fn_x2);
}

void eq_kernel_opencl(TensorIteratorBase& iter) {
  compare_eq_ne_kernel(iter, EqOpType::EQ);
}

void ne_kernel_opencl(TensorIteratorBase& iter) {
  compare_eq_ne_kernel(iter, EqOpType::NE);
}

REGISTER_DISPATCH(eq_stub, &eq_kernel_opencl)
REGISTER_DISPATCH(ne_stub, &ne_kernel_opencl)

} // namespace at::native
