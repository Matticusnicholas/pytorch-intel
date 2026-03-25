// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/EmbeddingBackwardKernel.cuh
#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>

namespace at::native {

Tensor embedding_backward_sycl_kernel(
    const Tensor &grad,
    const Tensor &orig_indices,
    const Tensor &sorted_indices,
    const Tensor &count,
    int64_t num_weights,
    int padding_idx = -1,
    bool mode_mean = false,
    const Tensor &offset2bag = Tensor(),
    const Tensor &bag_size = Tensor(),
    const Tensor &per_sample_weights = Tensor());

// Alias for backward compatibility
inline Tensor embedding_backward_cuda_kernel(
    const Tensor &grad,
    const Tensor &orig_indices,
    const Tensor &sorted_indices,
    const Tensor &count,
    int64_t num_weights,
    int padding_idx = -1,
    bool mode_mean = false,
    const Tensor &offset2bag = Tensor(),
    const Tensor &bag_size = Tensor(),
    const Tensor &per_sample_weights = Tensor()) {
  return embedding_backward_sycl_kernel(
      grad, orig_indices, sorted_indices, count, num_weights,
      padding_idx, mode_mean, offset2bag, bag_size, per_sample_weights);
}

} // namespace at::native
