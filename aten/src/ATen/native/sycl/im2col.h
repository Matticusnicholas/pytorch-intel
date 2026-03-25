// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/im2col.cuh
#pragma once

#include <ATen/AccumulateType.h>
#include <c10/macros/Macros.h>

namespace at::native {

// SYCL: im2col kernel uses sycl::nd_range parallel_for
// CUDA_KERNEL_LOOP_TYPE is replaced by computing global_id from nd_item
template <typename dt>
void im2col_sycl_kernel(
    sycl::nd_item<1> item,
    const int64_t n,
    const dt* data_im,
    const int64_t height,
    const int64_t width,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    const int64_t height_col,
    const int64_t width_col,
    dt* data_col) {
  auto index = item.get_global_id(0);
  auto stride = item.get_global_range(0);
  for (int64_t i = index; i < n; i += stride) {
    int64_t w_out = i % width_col;
    int64_t idx = i / width_col;
    int64_t h_out = idx % height_col;
    int64_t channel_in = idx / height_col;
    int64_t channel_out = channel_in * kernel_height * kernel_width;
    int64_t h_in = h_out * stride_height - pad_height;
    int64_t w_in = w_out * stride_width - pad_width;

    dt* col = data_col + (channel_out * height_col + h_out) * width_col + w_out;
    const dt* im = data_im + (channel_in * height + h_in) * width + w_in;

    for (int64_t ii = 0; ii < kernel_height; ++ii) {
      for (int64_t j = 0; j < kernel_width; ++j) {
        int64_t h = h_in + ii * dilation_height;
        int64_t w = w_in + j * dilation_width;
        *col = (h >= 0 && w >= 0 && h < height && w < width)
            ? im[ii * dilation_height * width + j * dilation_width]
            : static_cast<dt>(0);
        col += height_col * width_col;
      }
    }
  }
}

// SYCL: col2im kernel
template <typename dt, typename accT>
void col2im_sycl_kernel(
    sycl::nd_item<1> item,
    const int64_t n,
    const dt* data_col,
    const int64_t height,
    const int64_t width,
    const int64_t channels,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    const int64_t height_col,
    const int64_t width_col,
    dt* data_im) {
  auto index = item.get_global_id(0);
  auto global_stride = item.get_global_range(0);
  for (int64_t i = index; i < n; i += global_stride) {
    accT val = static_cast<accT>(0);
    const int64_t w_im = i % width + pad_w;
    const int64_t h_im = (i / width) % height + pad_h;
    const int64_t c_im = i / (width * height);
    int64_t kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int64_t kernel_extent_h = (kernel_h - 1) * dilation_h + 1;

    const int64_t w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int64_t w_col_end = std::min(w_im / stride_w + 1, width_col);
    const int64_t h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int64_t h_col_end = std::min(h_im / stride_h + 1, height_col);

    for (int64_t h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int64_t w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int64_t h_k = (h_im - h_col * stride_h);
        int64_t w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int64_t data_col_index =
              (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col) *
                  width_col + w_col;
          val += static_cast<accT>(data_col[data_col_index]);
        }
      }
    }
    data_im[i] = static_cast<dt>(val);
  }
}

} // namespace at::native
