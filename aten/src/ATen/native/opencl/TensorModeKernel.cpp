// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/TensorModeKernel.cpp

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/opencl/TensorModeKernel.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorCompare.h>

// SYCL subgroup size is typically 16 or 32 on Intel GPUs
constexpr int64_t MAX_BLOCK_SIZE = 256;

// Maximum size per grid dimension
constexpr int64_t MAX_GRID_SIZE = 65535LL;

namespace at::native {

void mode_kernel_impl(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto self_sizes = ensure_nonempty_vec(self.sizes().vec());
  int64_t ndim = ensure_nonempty_dim(self.dim());
  int64_t slice_size = ensure_nonempty_size(self, dim);
  int64_t slices = self.numel() / slice_size;

  assert(0 <= dim && static_cast<size_t>(dim) < self_sizes.size());
  self_sizes[dim] = 1;

  if (!keepdim) {
    if (values.ndimension() >= dim) {
      values.unsqueeze_(dim);
    }
    if (indices.ndimension() >= dim) {
      indices.unsqueeze_(dim);
    }
  }

  at::native::resize_output(values, self_sizes);
  at::native::resize_output(indices, self_sizes);

  if (slice_size == 1) {
    values.copy_(self);
    indices.fill_(0);
    if (!keepdim) {
      values.squeeze_(dim);
      indices.squeeze_(dim);
    }
    return;
  }

  auto transposed = self.transpose(dim, ndim - 1);
  auto contiguous = transposed.contiguous();

  auto values_transposed = values.transpose(dim, ndim - 1);
  auto indices_transposed = indices.transpose(dim, ndim - 1);

  if (slice_size <= 2 * MAX_BLOCK_SIZE &&
      slices <= MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE &&
      canUse32BitIndexMath(self)) {
    launch_fused_mode_kernel(
        values_transposed, indices_transposed, contiguous, slice_size, slices);
  } else {
    if (transposed.is_same(contiguous)) {
      contiguous = contiguous.clone();
    }

    launch_apply_mode_kernel(
        values_transposed, indices_transposed, contiguous, dim, ndim);
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
}

REGISTER_OPENCL_DISPATCH(mode_stub, &mode_kernel_impl)
} // namespace at::native
