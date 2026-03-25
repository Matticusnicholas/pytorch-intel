// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/fused_adagrad_impl.cuh
#pragma once
#include <ATen/core/Tensor.h>

namespace at::native {

void _fused_adagrad_sycl_impl_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const double lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf);

void _fused_adagrad_sycl_impl_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf);

// Aliases for backward compatibility
inline void _fused_adagrad_cuda_impl_(
    at::TensorList params, at::TensorList grads,
    at::TensorList state_sums, at::TensorList state_steps,
    const double lr, const double lr_decay, const double weight_decay,
    const double eps, const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  _fused_adagrad_sycl_impl_(params, grads, state_sums, state_steps,
      lr, lr_decay, weight_decay, eps, maximize, grad_scale, found_inf);
}

inline void _fused_adagrad_cuda_impl_(
    at::TensorList params, at::TensorList grads,
    at::TensorList state_sums, at::TensorList state_steps,
    const at::Tensor& lr, const double lr_decay, const double weight_decay,
    const double eps, const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  _fused_adagrad_sycl_impl_(params, grads, state_sums, state_steps,
      lr, lr_decay, weight_decay, eps, maximize, grad_scale, found_inf);
}

} // namespace at::native
