// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/fused_adam_amsgrad_impl.cu

#include <ATen/native/sycl/fused_adam_amsgrad_impl.h>

#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/sycl/MultiTensorApply.h>
#include <ATen/native/sycl/fused_adam_utils.h>
#include <vector>

namespace at::native {

void _fused_adam_amsgrad_opencl_impl_(
    at::TensorList params, at::TensorList grads,
    at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs, at::TensorList state_steps,
    const double lr, const double beta1, const double beta2,
    const double weight_decay, const double eps, const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params.vec(), grads.vec(), exp_avgs.vec(),
      exp_avg_sqs.vec(), max_exp_avg_sqs.vec()};

  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  const float* lr_ptr = nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, params[0].scalar_type(),
      "fused_adam_kernel_opencl", [&]() {
        multi_tensor_apply_for_fused_optimizer<5>(
            tensor_lists, state_steps,
            FusedAdamMathFunctor<scalar_t, 5, ADAM_MODE::ORIGINAL, true>(),
            lr_ptr, lr, beta1, beta2, weight_decay, eps, maximize,
            grad_scale_ptr, found_inf_ptr);
      });
}

void _fused_adam_amsgrad_opencl_impl_(
    at::TensorList params, at::TensorList grads,
    at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs, at::TensorList state_steps,
    const at::Tensor& lr, const double beta1, const double beta2,
    const double weight_decay, const double eps, const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params.vec(), grads.vec(), exp_avgs.vec(),
      exp_avg_sqs.vec(), max_exp_avg_sqs.vec()};

  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  const float* lr_ptr = lr.const_data_ptr<float>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, params[0].scalar_type(),
      "fused_adam_kernel_opencl", [&]() {
        multi_tensor_apply_for_fused_optimizer<5>(
            tensor_lists, state_steps,
            FusedAdamMathFunctor<scalar_t, 5, ADAM_MODE::ORIGINAL, true>(),
            lr_ptr, 1.0, beta1, beta2, weight_decay, eps, maximize,
            grad_scale_ptr, found_inf_ptr);
      });
}

} // namespace at::native
