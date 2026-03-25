// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/BinaryBitwiseOpsKernels.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

namespace at::native {

template<typename scalar_t>
struct BitwiseAndFunctor {
  inline scalar_t operator()(scalar_t a, scalar_t b) const {
    return a & b;
  }
};

template<>
struct BitwiseAndFunctor<bool> {
  inline bool operator()(bool a, bool b) const {
    return a && b;
  }
};

void bitwise_and_kernel_opencl(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_and_opencl", [&]() {
    BitwiseAndFunctor<scalar_t> f;
    opmath_symmetric_sycl_kernel_with_scalars<scalar_t>(iter, f);
  });
}

template<typename scalar_t>
struct BitwiseOrFunctor {
  inline scalar_t operator()(scalar_t a, scalar_t b) const {
    return a | b;
  }
};

template<>
struct BitwiseOrFunctor<bool> {
  inline bool operator()(bool a, bool b) const {
    return a || b;
  }
};

void bitwise_or_kernel_opencl(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_or_opencl", [&]() {
    BitwiseOrFunctor<scalar_t> f;
    opmath_symmetric_sycl_kernel_with_scalars<scalar_t>(iter, f);
  });
}

template<typename scalar_t>
struct BitwiseXorFunctor {
  inline scalar_t operator()(scalar_t a, scalar_t b) const {
    return a ^ b;
  }
};

template<>
struct BitwiseXorFunctor<bool> {
  inline bool operator()(bool a, bool b) const {
    return a != b;
  }
};

void bitwise_xor_kernel_opencl(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_xor_opencl", [&]() {
    BitwiseXorFunctor<scalar_t> f;
    opmath_symmetric_sycl_kernel_with_scalars<scalar_t>(iter, f);
  });
}

REGISTER_DISPATCH(bitwise_and_stub, &bitwise_and_kernel_opencl)
REGISTER_DISPATCH(bitwise_or_stub, &bitwise_or_kernel_opencl)
REGISTER_DISPATCH(bitwise_xor_stub, &bitwise_xor_kernel_opencl)

} // namespace at::native
