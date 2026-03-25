// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/AmpKernels.cu
// SYCL TODO: needs_review - complex kernel requires manual optimization for Intel GPU
// Contains multi_tensor_apply, raw CUDA kernel launch, and device-specific isfinite.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#define _USE_MATH_DEFINES

#include <math.h>

#include <ATen/core/Tensor.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/native/sycl/ForeachFunctors.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>

namespace {
static inline int isfinite_ensure_sycl_math(float val) {
  return std::isfinite(val);
}
}

namespace at::native {

namespace {
void _amp_non_finite_check_and_unscale_opencl_(Tensor& scaled_grad,
                                             Tensor& found_inf,
                                             const Tensor& inv_scale) {
  const OptionalDeviceGuard device_guard(device_of(scaled_grad));
  auto iter = TensorIterator::unary_op(scaled_grad, scaled_grad);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    iter.dtype(),
    "_amp_non_finite_check_and_unscale_opencl",
    [&iter, &found_inf, &inv_scale] {
      auto* found_inf_ptr = found_inf.mutable_data_ptr<float>();
      auto* inv_scale_ptr = inv_scale.const_data_ptr<float>();
      using opmath_t = at::opmath_type<scalar_t>;

      sycl_kernel(iter,
                 [found_inf_ptr, inv_scale_ptr] SYCL_LAMBDA (scalar_t val_in) -> scalar_t {
                   auto val = static_cast<opmath_t>(val_in);
                   if (!isfinite_ensure_sycl_math(val)) {
                     *found_inf_ptr = 1.f;
                   }
                   const auto inv_scale_val = *inv_scale_ptr;
                   return static_cast<scalar_t>(inv_scale_val == 1.f ? val : val * inv_scale_val);
                 });
    });
}
} // anonymous namespace

void _amp_foreach_non_finite_check_and_unscale_opencl_(TensorList scaled_grads,
                                                     Tensor& found_inf,
                                                     const Tensor& inv_scale) {
  if (scaled_grads.size() == 0) return;

  TORCH_CHECK(inv_scale.is_cuda(), "inv_scale must be a CUDA tensor.");
  TORCH_CHECK(found_inf.is_cuda(), "found_inf must be a CUDA tensor.");
  TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(inv_scale.scalar_type() == at::ScalarType::Float, "inv_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");

  check_foreach_api_restrictions(scaled_grads);

  // SYCL TODO: needs_review - multi_tensor_apply needs SYCL port
  // For now, fall back to per-tensor processing
  for (const Tensor& t : scaled_grads) {
    TORCH_CHECK(t.is_cuda(), "one of scaled_grads was not a CUDA tensor.");
    TORCH_CHECK(t.layout() == at::kStrided, "one of scaled_grads was not a strided tensor.");
    _amp_non_finite_check_and_unscale_opencl_(const_cast<Tensor&>(t), found_inf, inv_scale);
  }
}

// SYCL kernel for amp_update_scale
Tensor& _amp_update_scale_opencl_(Tensor& current_scale,
                                Tensor& growth_tracker,
                                const Tensor& found_inf,
                                double growth_factor,
                                double backoff_factor,
                                int64_t growth_interval) {
  TORCH_CHECK(growth_tracker.is_cuda(), "growth_tracker must be a CUDA tensor.");
  TORCH_CHECK(current_scale.is_cuda(), "current_scale must be a CUDA tensor.");
  TORCH_CHECK(found_inf.is_cuda(), "found_inf must be a CUDA tensor.");
  TORCH_CHECK(growth_tracker.numel() == 1, "growth_tracker must be a 1-element tensor.");
  TORCH_CHECK(current_scale.numel() == 1, "current_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(growth_tracker.scalar_type() == at::ScalarType::Int, "growth_tracker must be an int tensor.");
  TORCH_CHECK(current_scale.scalar_type() == at::ScalarType::Float, "current_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");

  auto& queue = at::sycl::getCurrentSYCLQueue();
  auto* current_scale_ptr = current_scale.mutable_data_ptr<float>();
  auto* growth_tracker_ptr = growth_tracker.mutable_data_ptr<int>();
  auto* found_inf_ptr = found_inf.const_data_ptr<float>();

  queue.single_task([=]() {
    if (*found_inf_ptr) {
      *current_scale_ptr = (*current_scale_ptr) * backoff_factor;
      *growth_tracker_ptr = 0;
    } else {
      auto successful = (*growth_tracker_ptr) + 1;
      if (successful == growth_interval) {
        auto new_scale = static_cast<float>((*current_scale_ptr) * growth_factor);
        if (std::isfinite(new_scale)) {
          *current_scale_ptr = new_scale;
        }
        *growth_tracker_ptr = 0;
      } else {
        *growth_tracker_ptr = successful;
      }
    }
  });

  return current_scale;
}

} // namespace at::native
