// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/TensorTopK.cpp

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/opencl/TensorTopK.h>

#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/opencl/Sort.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/OpenCLFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/sort_opencl_dispatch.h>
#include <ATen/ops/topk_native.h>
#endif

namespace at::native {

void topk_out_with_sort(
  const Tensor& self,
  int64_t k, int64_t dim, bool largest,
  const Tensor& values,
  const Tensor& indices
) {
  auto [sorted_values, sorted_indices] = at::opencl::sort(self, /* stable= */false, dim, largest);
  values.copy_(sorted_values.narrow(dim, 0, k));
  indices.copy_(sorted_indices.narrow(dim, 0, k));
}

bool should_use_sort(const Tensor& self, int64_t dim) {
  // Intel GPUs do not have the same ROCm-specific considerations.
  // For now, default to the radix-select TopK path.
  return false;
}

TORCH_IMPL_FUNC(topk_out_opencl)
  (const Tensor& self,
   int64_t k, int64_t dim, bool largest, bool sorted,
   const Tensor& values,
   const Tensor& indices) {
  TensorArg topK_arg{values, "topK", 1}, indices_arg{indices, "indices", 2}, input_arg{self, "self", 3};
  checkAllSameGPU(__func__, {topK_arg, indices_arg, input_arg});

  dim = at::maybe_wrap_dim(dim, self);

  if (should_use_sort(self, dim)) {
    topk_out_with_sort(self, k, dim, largest, values, indices);
    return;
  }

  if (k == 0) {
    return;
  }

  launch_gather_topk_kernel(self, k, dim, largest, values, indices);

  if (sorted && values.numel() > 1) {
    if (should_use_small_sort(values, dim)) {
      sortKeyValueInplace(values, indices, dim, largest);
    } else {
      Tensor sortedIndices = at::empty_like(indices);
      Tensor sortedValues = at::empty_like(values);
      at::opencl::sort_outf(values, /* stable= */ false, dim, largest, sortedValues, sortedIndices);
      indices.copy_(indices.gather(dim, sortedIndices));
      values.copy_(sortedValues);
    }
  }
}

} // namespace at::native
