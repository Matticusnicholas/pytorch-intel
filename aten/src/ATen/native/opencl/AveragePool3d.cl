// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/AveragePool3d.cu
// SYCL TODO: needs_review - complex kernel requires manual optimization for Intel GPU
// Contains multiple raw CUDA kernels, PackedTensorAccessor, atomicAdd, <<<>>> launch.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Pool.h>
#include <ATen/sycl/Atomic.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/sycl/detail/TensorInfo.h>
#include <ATen/sycl/detail/IndexUtils.h>
#include <ATen/sycl/detail/KernelUtils.h>
#include <ATen/native/sycl/KernelUtils.h>
#include <c10/macros/Macros.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/avg_pool3d_native.h>
#include <ATen/ops/avg_pool3d_backward_native.h>
#endif

namespace at::native {

// SYCL TODO: needs_review - This file contains multiple complex CUDA kernels
// (avg_pool3d_cuda_update_output, avg_pool3d_single_backward_out_frame_stride1,
// avg_pool3d_cuda_update_grad_input_atomic, avg_pool3d_cuda_update_grad_input)
// that use PackedTensorAccessor64, dim3 grid/block, and fastAtomicAdd.
// Full translation requires extensive manual SYCL kernel rewrite.

TORCH_IMPL_FUNC(avg_pool3d_out_opencl)(
  const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride,
  IntArrayRef padding, bool ceil_mode, bool count_include_pad,
  std::optional<int64_t> divisor_override, const Tensor& output) {
  TORCH_CHECK(false, "avg_pool3d_out_opencl: full implementation pending SYCL port of complex kernels");
}

TORCH_IMPL_FUNC(avg_pool3d_backward_out_opencl)(
  const Tensor& gradOutput, const Tensor& input,
  IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding,
  bool ceil_mode, bool count_include_pad,
  std::optional<int64_t> divisor_override, const Tensor& gradInput) {
  globalContext().alertNotDeterministic("avg_pool3d_backward_opencl");
  TORCH_CHECK(false, "avg_pool3d_backward_out_opencl: full implementation pending SYCL port of complex kernels");
}

} // namespace at::native
