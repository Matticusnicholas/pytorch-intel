// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/GridSampler.cuh
#pragma once
#include <ATen/native/sycl/KernelUtils.h>
#include <ATen/native/GridSamplerUtils.h>
#include <cmath>

namespace at::native {

using detail::GridSamplerInterpolation;
using detail::GridSamplerPadding;

template <typename scalar_t>
inline
scalar_t grid_sampler_unnormalize(scalar_t coord, int size, bool align_corners) {
  if (align_corners) {
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    return ((coord + 1.f) * size - 1) / 2;
  }
}

template <typename scalar_t>
inline
scalar_t grid_sampler_unnormalize_set_grad(scalar_t coord, int size,
                                           bool align_corners, scalar_t *grad_in) {
  if (align_corners) {
    *grad_in = static_cast<scalar_t>(size - 1) / 2;
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    *grad_in = static_cast<scalar_t>(size) / 2;
    return ((coord + 1.f) * size - 1) / 2;
  }
}

template <typename scalar_t>
inline
scalar_t clip_coordinates(scalar_t in, int clip_limit) {
  return std::min(static_cast<scalar_t>(clip_limit - 1), std::max(in, static_cast<scalar_t>(0)));
}

template <typename scalar_t>
inline
scalar_t clip_coordinates_set_grad(scalar_t in, int clip_limit, scalar_t *grad_in) {
  if (in <= static_cast<scalar_t>(0)) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  } else {
    scalar_t max = static_cast<scalar_t>(clip_limit - 1);
    if (in >= max) {
      *grad_in = static_cast<scalar_t>(0);
      return max;
    } else {
      *grad_in = static_cast<scalar_t>(1);
      return in;
    }
  }
}

template <typename scalar_t>
inline
scalar_t reflect_coordinates(scalar_t in, int twice_low, int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = std::fabs(in - min);
  scalar_t extra = std::fmod(in, span);
  int flips = static_cast<int>(std::floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

template <typename scalar_t>
inline
scalar_t reflect_coordinates_set_grad(scalar_t in, int twice_low, int twice_high,
                                      scalar_t *grad_in) {
  if (twice_low == twice_high) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = in - min;
  if (in < static_cast<scalar_t>(0)) {
    *grad_in = static_cast<scalar_t>(-1);
    in = -in;
  } else {
    *grad_in = static_cast<scalar_t>(1);
  }
  scalar_t extra = std::fmod(in, span);
  int flips = static_cast<int>(std::floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    *grad_in = -(*grad_in);
    return span - extra + min;
  }
}

template <typename scalar_t>
inline
scalar_t safe_downgrade_to_int_range(scalar_t x) {
  if (x > INT_MAX - 1 || x < INT_MIN || !std::isfinite(static_cast<double>(x)))
    return static_cast<scalar_t>(-100.0);
  return x;
}

template<typename scalar_t>
inline
scalar_t compute_coordinates(scalar_t coord, int size,
                             GridSamplerPadding padding_mode,
                             bool align_corners) {
  if (padding_mode == GridSamplerPadding::Border) {
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2 * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2 * size - 1);
    }
    coord = clip_coordinates(coord, size);
  }
  return coord;
}

template<typename scalar_t>
inline
scalar_t grid_sampler_compute_source_index(
    scalar_t coord, int size, GridSamplerPadding padding_mode,
    bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  coord = compute_coordinates(coord, size, padding_mode, align_corners);
  return coord;
}

template<typename scalar_t>
inline
scalar_t grid_sampler_compute_source_index_set_grad(
    scalar_t coord, int size, GridSamplerPadding padding_mode,
    bool align_corners, scalar_t *grad_in) {
  scalar_t grad_clip, grad_refl;
  coord = grid_sampler_unnormalize_set_grad(coord, size, align_corners, grad_in);
  if (padding_mode == GridSamplerPadding::Border) {
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_clip;
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    if (align_corners) {
      coord = reflect_coordinates_set_grad(coord, 0, 2 * (size - 1), &grad_refl);
    } else {
      coord = reflect_coordinates_set_grad(coord, -1, 2 * size - 1, &grad_refl);
    }
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_refl * grad_clip;
  }
  return coord;
}

} // namespace at::native
