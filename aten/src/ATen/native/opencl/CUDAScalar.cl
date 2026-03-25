// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/CUDAScalar.cu
// SYCL TODO: needs_review - complex kernel requires manual optimization for Intel GPU
// Contains cudaMemcpy, cudaStream, pinned memory allocation patterns.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/EmptyTensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_local_scalar_dense_native.h>
#endif

#include <ATen/sycl/SYCLContext.h>

namespace at::native {

Scalar _local_scalar_dense_opencl(const Tensor& self) {
  Scalar r;
  TORCH_CHECK(self.numel() > 0, "_local_scalar_dense: Empty tensor not supported");
    AT_DISPATCH_V2(
      self.scalar_type(), "_local_scalar_dense_opencl", AT_WRAP([&] {
          auto value = at::detail::empty_cpu(
            {1},
            c10::CppTypeToScalarType<scalar_t>(),
            std::nullopt, std::nullopt, true, std::nullopt);
          // SYCL: use queue.memcpy for device-to-host transfer
          auto& queue = at::sycl::getCurrentSYCLQueue();
          queue.memcpy(value.mutable_data_ptr<scalar_t>(),
                       self.const_data_ptr<scalar_t>(),
                       sizeof(scalar_t));
          queue.wait();
          r = Scalar(*value.const_data_ptr<scalar_t>());
        }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kHalf, kBool, kBFloat16, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  return r;
}

} // at::native
