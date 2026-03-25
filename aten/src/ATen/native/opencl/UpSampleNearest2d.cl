// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/UpSampleNearest2d.cu

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/native/sycl/LaunchUtils.h>
#include <ATen/native/sycl/UpSample.h>
#include <ATen/native/sycl/KernelUtils.h>
#include <ATen/sycl/detail/KernelUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_nearest_exact2d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact2d_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/upsample_nearest2d_backward_native.h>
#include <ATen/ops/upsample_nearest2d_native.h>
#endif

namespace at::native {
namespace {

#define MAX_THREADS 512

typedef int (*nn_compute_source_index_fn_t)(const float, int, int);
typedef int (*nn_bw_compute_source_index_fn_t)(const float, int, int);

template <typename scalar_t, nn_compute_source_index_fn_t nn_compute_source_index_fn>
void upsample_nearest2d_out_frame_kernel(
    sycl::nd_item<3> item,
    const scalar_t* idata,
    scalar_t* odata,
    const size_t nc,
    const size_t height1,
    const size_t width1,
    const size_t height2,
    const size_t width2,
    float height_scale,
    float width_scale) {
  size_t nc_iter = item.get_local_id(2) + item.get_group(2) * item.get_local_range(2);
  int64_t w2 = ((int64_t) item.get_local_id(0)) + item.get_group(0) * item.get_local_range(0);
  int64_t h2 = item.get_local_id(1) + item.get_group(1) * item.get_local_range(1);

  if (w2 >= width2 || h2 >= height2) {
    return;
  }

  int64_t nc_stride = ((int64_t) item.get_local_range(2)) * item.get_group_range(2);

  const size_t h1 = height1 == height2
      ? h2
      : nn_compute_source_index_fn(height_scale, h2, height1);
  const size_t w1 = width1 == width2
      ? w2
      : nn_compute_source_index_fn(width_scale, w2, width1);

  size_t src_index = (nc_iter * height1 + h1) * width1 + w1;
  size_t src_index_stride = nc_stride * width1 * height1;
  size_t dst_index = (nc_iter * height2 + h2) * width2 + w2;
  size_t dst_index_stride = nc_stride * width2 * height2;

  while (nc_iter < nc) {
    odata[dst_index] = idata[src_index];
    dst_index += dst_index_stride;
    src_index += src_index_stride;
    nc_iter += nc_stride;
  }
}

template <typename scalar_t, nn_compute_source_index_fn_t nn_compute_source_index_fn>
void upsample_nearest2d_nhwc_out_frame_kernel(
    sycl::nd_item<1> item,
    const scalar_t* idata,
    scalar_t* odata,
    const size_t channels,
    const size_t height1,
    const size_t width1,
    const size_t height2,
    const size_t width2,
    float height_scale,
    float width_scale,
    const size_t out_numel) {

    const int64_t index = ((int64_t) item.get_group(0)) * item.get_local_range(0) + item.get_local_id(0);

    if (index < out_numel) {
    const auto c = index % channels;
    const auto w2 = (index / channels) % width2;
    const auto h2 = (index / channels / width2) % height2;
    const auto n = index / channels / width2 / height2;

    const size_t h1 = height1 == height2 ? h2 : nn_compute_source_index_fn(height_scale, h2, height1);
    const size_t w1 = width1 == width2 ? w2 : nn_compute_source_index_fn(width_scale, w2, width1);

    odata[index] = idata[idx_cl(n, h1, w1, c, height1, width1, channels)];
  }
}

template <typename scalar_t, typename accscalar_t, nn_bw_compute_source_index_fn_t nn_bw_compute_source_index_fn>
void upsample_nearest2d_backward_out_frame_kernel(
    sycl::nd_item<1> item,
    const scalar_t* grad_o,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_h,
    size_t src_dim_w,
    size_t dst_dim_h,
    size_t dst_dim_w,
    scalar_t* grad_i,
    float height_scale,
    float width_scale) {
  int64_t dst_idx = ((int64_t) item.get_group(0)) * item.get_local_range(0) + item.get_local_id(0);
  if (dst_idx >= dim_c * dst_dim_h * dst_dim_w)
    return;

  int64_t dst_c_stride = dst_dim_h * dst_dim_w;
  int64_t src_c_stride = src_dim_h * src_dim_w;

  int c = (dst_idx / (dst_c_stride)) % dim_c;

  int dst_y = (dst_idx / dst_dim_w) % dst_dim_h;
  int src_y = nn_bw_compute_source_index_fn(height_scale, dst_y, src_dim_h);
  int src_y_up = nn_bw_compute_source_index_fn(height_scale, dst_y + 1, src_dim_h);

  int dst_x = dst_idx % dst_dim_w;
  int src_x = nn_bw_compute_source_index_fn(width_scale, dst_x, src_dim_w);
  int src_x_up = nn_bw_compute_source_index_fn(width_scale, dst_x + 1, src_dim_w);

  for (int b = 0; b < dim_b; b++) {
    accscalar_t grad = 0;
    for (int y = src_y; y < src_y_up; y++) {
      for (int x = src_x; x < src_x_up; x++) {
        int64_t src_idx =
            b * dim_c * src_c_stride + c * src_c_stride + y * src_dim_w + x;
        grad += grad_o[src_idx];
      }
    }
    grad_i[dst_idx] = grad;
    dst_idx += dim_c * dst_c_stride;
  }
}

template <typename scalar_t, typename accscalar_t, nn_bw_compute_source_index_fn_t nn_bw_compute_source_index_fn>
void upsample_nearest2d_backward_nhwc_out_frame_kernel(
    sycl::nd_item<1> item,
    const scalar_t* go,
    scalar_t* gi,
    const size_t height1,
    const size_t width1,
    const size_t height2,
    const size_t width2,
    const size_t channels,
    const float height_scale,
    const float width_scale,
    const size_t gi_numel) {

  const int64_t index = ((int64_t) item.get_group(0)) * item.get_local_range(0) + item.get_local_id(0);

  if (index < gi_numel) {
    const int c = index % channels;
    const int w2 = (index / channels) % width2;
    const int h2 = (index / channels / width2) % height2;
    const int n = index / channels / width2 / height2;

    int h1 = nn_bw_compute_source_index_fn(height_scale, h2, height1);
    int h1_up = nn_bw_compute_source_index_fn(height_scale, h2 + 1, height1);

    int w1 = nn_bw_compute_source_index_fn(width_scale, w2, width1);
    int w1_up = nn_bw_compute_source_index_fn(width_scale, w2 + 1, width1);

    accscalar_t grad = 0;
    for (int ih = h1; ih < h1_up; ih++) {
      for (int iw = w1; iw < w1_up; iw++) {
        grad += go[idx_cl(n, ih, iw, c, height1, width1, channels)];
      }
    }
    gi[index] = static_cast<scalar_t>(grad);
  }
}

template<nn_compute_source_index_fn_t nn_compute_source_index_fn>
static void upsample_nearest2d_out_opencl_template(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg input_arg{input_, "input_", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  if (input_.numel() == 0) {
    return;
  }

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_height = input_.size(2);
  int input_width = input_.size(3);

  const float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
  const float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

  const auto memory_format = input_.suggest_memory_format();

  if (input_.sizes() == output.sizes()) {
    output.copy_(input_);
    return;
  }

  if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 4 &&
        output.is_contiguous(memory_format)) {
    at::Tensor input = input_.contiguous(at::MemoryFormat::ChannelsLast);

    TORCH_CHECK(input.numel() < std::numeric_limits<int64_t>::max(),
      "upsample_nearest_nhwc only supports input tensors with less than 2^63 - 1 elements");
    TORCH_CHECK(output.numel() < std::numeric_limits<int64_t>::max(),
      "upsample_nearest_nhwc only supports output tensors with less than 2^63 - 1 elements");

    const int64_t num_kernels = output.numel();
    const int64_t num_threads = std::min<int64_t>(1024, 1024);

    auto stream = at::sycl::getCurrentSYCLStream();
    AT_DISPATCH_FLOATING_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Byte, input.scalar_type(), "upsample_nearest2d_nhwc_out_frame", [&] {
      const scalar_t* idata = input.const_data_ptr<scalar_t>();
      scalar_t* odata = output.mutable_data_ptr<scalar_t>();
      int64_t groups = ceil_div(num_kernels, num_threads);
      stream.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(groups * num_threads, num_threads),
            [=](sycl::nd_item<1> item) {
              upsample_nearest2d_nhwc_out_frame_kernel<scalar_t, nn_compute_source_index_fn>(
                  item, idata, odata, channels, input_height, input_width,
                  output_height, output_width, height_scale, width_scale, num_kernels);
            });
      });
    });
  } else {
    Tensor output_c = output.is_contiguous() ? output : at::empty(output.sizes(), output.options());
    Tensor input = input_.contiguous();

    int64_t nc = nbatch * channels;
    const int max_threads = MAX_THREADS;

    int block_x = std::min<int>(lastPow2(output_width), max_threads);
    int block_y = std::min<int>(lastPow2(output_height), max_threads / block_x);
    int block_z = std::min<int>((int)nc, max_threads / block_x / block_y);

    int grid_x = ceil_div(output_width, block_x);
    int grid_y = ceil_div(output_height, block_y);
    int grid_z = std::min<int>(65535, ceil_div((int)nc, (int)(block_z * 4)));

    auto stream = at::sycl::getCurrentSYCLStream();
    AT_DISPATCH_FLOATING_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Byte, input.scalar_type(), "upsample_nearest2d_out_frame", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          auto idata = input.const_data_ptr<scalar_t>();
          auto odata = output_c.mutable_data_ptr<scalar_t>();

          stream.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(grid_x * block_x, grid_y * block_y, grid_z * block_z),
                    sycl::range<3>(block_x, block_y, block_z)),
                [=](sycl::nd_item<3> item) {
                  upsample_nearest2d_out_frame_kernel<scalar_t, nn_compute_source_index_fn>(
                      item, idata, odata, nc, input_height, input_width,
                      output_height, output_width, height_scale, width_scale);
                });
          });
        });

    if (!output.is_contiguous()) {
        output.copy_(output_c);
    }
  }
}

template<nn_bw_compute_source_index_fn_t nn_bw_compute_source_index_fn>
static void upsample_nearest2d_backward_out_opencl_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(__func__, {grad_output_arg, grad_input_arg});

  if (grad_input.numel() == 0) {
    return;
  }

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_height = input_size[2];
  int input_width = input_size[3];

  const float height_scale = compute_scales_value_backwards<float>(scales_h, output_height, input_height);
  const float width_scale = compute_scales_value_backwards<float>(scales_w, output_width, input_width);

  auto memory_format = grad_output_.suggest_memory_format();

  if (grad_output_.sizes() == grad_input.sizes()) {
    grad_input.copy_(grad_output_);
    return;
  }

  if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 4 &&
        grad_input.is_contiguous(memory_format)) {
    Tensor grad_output = grad_output_.contiguous(at::MemoryFormat::ChannelsLast);

    const int num_kernels = grad_input.numel();
    const int num_threads = std::min(1024, 1024);

    auto stream = at::sycl::getCurrentSYCLStream();
    AT_DISPATCH_FLOATING_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Byte, grad_output.scalar_type(), "upsample_nearest2d_backward_nhwc_out_frame", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      const scalar_t* go = grad_output.const_data_ptr<scalar_t>();
      scalar_t* gi = grad_input.mutable_data_ptr<scalar_t>();

      int groups = ceil_div(num_kernels, num_threads);
      stream.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(groups * num_threads, num_threads),
            [=](sycl::nd_item<1> item) {
              upsample_nearest2d_backward_nhwc_out_frame_kernel<scalar_t, accscalar_t, nn_bw_compute_source_index_fn>(
                  item, go, gi, output_height, output_width, input_height, input_width,
                  channels, height_scale, width_scale, num_kernels);
            });
      });
    });
  } else {
    Tensor grad_input_c = grad_input.is_contiguous() ? grad_input : at::empty(grad_input.sizes(), grad_input.options());
    Tensor grad_output = grad_output_.contiguous();

    size_t n = grad_input.numel() / nbatch;
    unsigned int bdim = std::min<unsigned int>(MAX_THREADS, MAX_THREADS);
    unsigned int gdim = (unsigned int) ceil_div(n, (size_t) bdim);

    auto stream = at::sycl::getCurrentSYCLStream();
    AT_DISPATCH_FLOATING_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Byte, grad_output.scalar_type(), "upsample_nearest2d_backward_out_frame", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      auto idata = grad_input_c.mutable_data_ptr<scalar_t>();
      auto odata = grad_output.const_data_ptr<scalar_t>();

      stream.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(gdim * bdim, bdim),
            [=](sycl::nd_item<1> item) {
              upsample_nearest2d_backward_out_frame_kernel<scalar_t, accscalar_t, nn_bw_compute_source_index_fn>(
                  item, odata, nbatch, channels, output_height, output_width,
                  input_height, input_width, idata, height_scale, width_scale);
            });
      });
    });

    if (!grad_input.is_contiguous()) {
        grad_input.copy_(grad_input_c);
    }
  }
}

} // namespace

TORCH_IMPL_FUNC(upsample_nearest2d_out_opencl) (
    const Tensor& input, IntArrayRef output_size,
    std::optional<double> scales_h, std::optional<double> scales_w,
    const Tensor& output) {
  upsample_nearest2d_out_opencl_template<nearest_neighbor_compute_source_index>(
      output, input, output_size, scales_h, scales_w);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_out_opencl) (
    const Tensor& input, IntArrayRef output_size,
    std::optional<double> scales_h, std::optional<double> scales_w,
    const Tensor& output) {
  upsample_nearest2d_out_opencl_template<nearest_neighbor_exact_compute_source_index>(
      output, input, output_size, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_opencl) (
    const Tensor& grad_output, IntArrayRef output_size,
    IntArrayRef input_size, std::optional<double> scales_h,
    std::optional<double> scales_w, const Tensor& grad_input) {
  upsample_nearest2d_backward_out_opencl_template<nearest_neighbor_bw_compute_source_index>(
      grad_input, grad_output, output_size, input_size, scales_h, scales_w);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_backward_out_opencl) (
    const Tensor& grad_output, IntArrayRef output_size,
    IntArrayRef input_size, std::optional<double> scales_h,
    std::optional<double> scales_w, const Tensor& grad_input) {
  upsample_nearest2d_backward_out_opencl_template<nearest_neighbor_exact_bw_compute_source_index>(
      grad_input, grad_output, output_size, input_size, scales_h, scales_w);
}

} // namespace at::native
