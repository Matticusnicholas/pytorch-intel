// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/UpSample.cuh
#pragma once
#include <ATen/core/TensorAccessor.h>

#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <c10/util/OptionalArrayRef.h>

#include <math.h>
#include <optional>

namespace at::native {

namespace upsample {
TORCH_API c10::SmallVector<int64_t, 3> compute_output_size(
    c10::IntArrayRef input_size,
    at::OptionalIntArrayRef output_size,
    std::optional<c10::ArrayRef<double>> scale_factors);
} // namespace upsample

namespace upsample_sycl {

inline std::optional<double> get_scale_value(std::optional<c10::ArrayRef<double>> scales, int idx) {
  if (!scales) {
    return std::nullopt;
  }
  return scales->at(idx);
}

} // namespace upsample_sycl

// Backward-compatible alias
namespace upsample_cuda = upsample_sycl;

template <typename scalar_t>
inline scalar_t min(scalar_t a, scalar_t b) {
  return a < b ? a : b;
}

template <typename scalar_t>
inline scalar_t max(scalar_t a, scalar_t b) {
  return a > b ? a : b;
}

template <typename accscalar_t>
inline accscalar_t compute_scales_value(
    const std::optional<double> scale,
    int64_t src_size,
    int64_t dst_size) {
  return (scale.has_value() && scale.value() > 0.) ? (accscalar_t)(1.0 / scale.value())
                                                   : (accscalar_t)src_size / dst_size;
}

template <typename accscalar_t>
inline accscalar_t compute_scales_value_backwards(
    const std::optional<double> scale,
    int64_t src_size,
    int64_t dst_size) {
  return (scale.has_value() && scale.value() > 0.) ? (accscalar_t)scale.value()
                                                   : (accscalar_t)dst_size / src_size;
}

template <typename accscalar_t>
inline accscalar_t area_pixel_compute_scale(
    int64_t input_size,
    int64_t output_size,
    bool align_corners,
    const std::optional<double> scale) {
  if (align_corners) {
    if (output_size > 1) {
      return (accscalar_t)(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<accscalar_t>(0);
    }
  } else {
    return compute_scales_value<accscalar_t>(scale, input_size, output_size);
  }
}

template <typename accscalar_t>
inline accscalar_t area_pixel_compute_source_index(
    accscalar_t scale,
    int64_t dst_index,
    bool align_corners,
    bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    accscalar_t src_idx = scale * (dst_index + static_cast<accscalar_t>(0.5)) -
        static_cast<accscalar_t>(0.5);
    return (!cubic && src_idx < static_cast<accscalar_t>(0))
        ? static_cast<accscalar_t>(0)
        : src_idx;
  }
}

} // namespace at::native
