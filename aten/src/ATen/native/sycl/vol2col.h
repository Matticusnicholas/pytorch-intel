// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/vol2col.cuh
#pragma once

#include <c10/macros/Macros.h>

namespace at::native {

// SYCL: vol2col kernel uses sycl::nd_range parallel_for
template <typename T>
void vol2col_sycl_kernel(
    sycl::nd_item<1> item,
    const int64_t n,
    const T* data_vol,
    const int depth,
    const int height,
    const int width,
    const int ksize_t,
    const int ksize_h,
    const int ksize_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int depth_col,
    const int height_col,
    const int width_col,
    T* data_col) {
  auto index = item.get_global_id(0);
  auto global_stride = item.get_global_range(0);
  for (int64_t idx = index; idx < n; idx += global_stride) {
    auto local_index = idx;
    auto w_out = local_index % width_col;
    local_index /= width_col;
    auto h_out = local_index % height_col;
    local_index /= height_col;
    auto t_out = local_index % depth_col;
    auto channel_in = local_index / depth_col;
    auto channel_out = channel_in * ksize_t * ksize_h * ksize_w;
    auto t_in = t_out * stride_t - pad_t;
    auto h_in = h_out * stride_h - pad_h;
    auto w_in = w_out * stride_w - pad_w;
    T* col_ptr = data_col +
        ((channel_out * depth_col + t_out) * height_col + h_out) * width_col +
        w_out;
    const T* vol_ptr = data_vol +
        ((channel_in * depth + t_in) * height + h_in) * width + w_in;
    for (int i = 0; i < ksize_t; ++i) {
      for (int j = 0; j < ksize_h; ++j) {
        for (int k = 0; k < ksize_w; ++k) {
          auto t = t_in + i * dilation_t;
          auto h = h_in + j * dilation_h;
          auto w = w_in + k * dilation_w;
          *col_ptr = (t >= 0 && h >= 0 && w >= 0 && t < depth && h < height &&
                      w < width)
              ? vol_ptr
                    [i * dilation_t * height * width + j * dilation_h * width +
                     k * dilation_w]
              : static_cast<T>(0);
          col_ptr += depth_col * height_col * width_col;
        }
      }
    }
  }
}

// SYCL: col2vol kernel
template <typename T, typename accT>
void col2vol_sycl_kernel(
    sycl::nd_item<1> item,
    const int64_t n,
    const T* data_col,
    const int depth,
    const int height,
    const int width,
    const int channels,
    const int ksize_t,
    const int ksize_h,
    const int ksize_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int depth_col,
    const int height_col,
    const int width_col,
    T* data_vol) {
  auto index = item.get_global_id(0);
  auto global_stride = item.get_global_range(0);
  for (int64_t i = index; i < n; i += global_stride) {
    accT val = static_cast<accT>(0);
    const int w_im = i % width + pad_w;
    const int h_im = (i / width) % height + pad_h;
    const int t_im = (i / width / height) % depth + pad_t;
    const int c_im = i / (width * height * depth);

    int kernel_extent_w = (ksize_w - 1) * dilation_w + 1;
    int kernel_extent_h = (ksize_h - 1) * dilation_h + 1;
    int kernel_extent_t = (ksize_t - 1) * dilation_t + 1;

    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = std::min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = std::min(h_im / stride_h + 1, height_col);
    const int t_col_start =
        (t_im < kernel_extent_t) ? 0 : (t_im - kernel_extent_t) / stride_t + 1;
    const int t_col_end = std::min(t_im / stride_t + 1, depth_col);

    for (int t_col = t_col_start; t_col < t_col_end; t_col += 1) {
      for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
        for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
          int t_k = (t_im - t_col * stride_t);
          int h_k = (h_im - h_col * stride_h);
          int w_k = (w_im - w_col * stride_w);
          if (t_k % dilation_t == 0 && h_k % dilation_h == 0 &&
              w_k % dilation_w == 0) {
            t_k /= dilation_t;
            h_k /= dilation_h;
            w_k /= dilation_w;
            int data_col_index =
                (((((c_im * ksize_t + t_k) * ksize_h + h_k) * ksize_w + w_k) *
                     depth_col + t_col) * height_col + h_col) *
                    width_col + w_col;
            val += static_cast<accT>(data_col[data_col_index]);
          }
        }
      }
    }
    data_vol[i] = static_cast<T>(val);
  }
}

} // namespace at::native
