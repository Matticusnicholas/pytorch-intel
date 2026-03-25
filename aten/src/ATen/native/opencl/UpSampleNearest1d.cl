// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/UpSampleNearest1d.cu

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/native/sycl/UpSample.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/upsample_nearest1d_native.h>
#include <ATen/ops/upsample_nearest1d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact1d_native.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward_native.h>
#endif

namespace at::native {
namespace {

#define MAX_THREADS 512

typedef int (*nn_compute_source_index_fn_t)(const float, int, int);
typedef int (*nn_bw_compute_source_index_fn_t)(const float, int, int);

template <typename scalar_t, nn_compute_source_index_fn_t nn_compute_source_index_fn>
void upsample_nearest1d_out_frame_kernel(
    sycl::nd_item<1> item,
    const scalar_t* input,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_w,
    size_t dst_dim_w,
    scalar_t* output,
    float scale_factor) {
  int dst_idx = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  if (dst_idx >= dim_c * dst_dim_w)
    return;

  int c = (dst_idx / dst_dim_w) % dim_c;
  int dst_x = dst_idx % dst_dim_w;
  int src_x = nn_compute_source_index_fn(scale_factor, dst_x, src_dim_w);

  int src_idx = c * src_dim_w + src_x;
  int src_stride = dim_c * src_dim_w;
  int dst_stride = dim_c * dst_dim_w;

  for (int b = 0; b < dim_b; b++) {
    output[dst_idx] = input[src_idx];
    src_idx += src_stride;
    dst_idx += dst_stride;
  }
}

template <typename scalar_t, typename accscalar_t, nn_bw_compute_source_index_fn_t nn_bw_compute_source_index_fn>
void upsample_nearest1d_backward_out_frame_kernel(
    sycl::nd_item<1> item,
    const scalar_t* grad_o,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_w,
    size_t dst_dim_w,
    scalar_t* grad_i,
    float scale_factor) {

  int dst_idx = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  if (dst_idx >= dim_c * dst_dim_w)
    return;

  int c = (dst_idx / (dst_dim_w)) % dim_c;
  int dst_x = dst_idx % dst_dim_w;
  int src_x = nn_bw_compute_source_index_fn(scale_factor, dst_x, src_dim_w);
  int src_x_up = nn_bw_compute_source_index_fn(scale_factor, dst_x+1, src_dim_w);

  for (int b = 0; b < dim_b; b++) {
    accscalar_t grad = 0;
    int src_idx = b * dim_c * src_dim_w + c * src_dim_w + src_x;
    for (int x = src_x; x < src_x_up; x++) {
      grad += grad_o[src_idx++];
    }
    grad_i[dst_idx] = grad;
    dst_idx += dim_c * dst_dim_w;
  }
}

template<nn_compute_source_index_fn_t nn_compute_source_index_fn>
static void upsample_nearest1d_out_opencl_template(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    std::optional<double> scales) {
  TensorArg input_arg{input_, "input_", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_nearest1d_out_opencl", {input_arg, output_arg});

  int output_width = output_size[0];
  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_width = input_.size(2);

  Tensor input = input_.contiguous();

  if (input.numel() == 0) {
    return;
  }

  unsigned int n = output.numel() / nbatch;
  unsigned int bdim = std::min<unsigned int>(MAX_THREADS, MAX_THREADS);
  unsigned int gdim = ceil_div(n, bdim);
  TORCH_CHECK(output.numel() <= std::numeric_limits<int32_t>::max());

  auto stream = at::sycl::getCurrentSYCLStream();
  AT_DISPATCH_FLOATING_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Byte, input.scalar_type(), "upsample_nearest1d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.const_data_ptr<scalar_t>();
        auto odata = output.mutable_data_ptr<scalar_t>();

        const float scale_factor = compute_scales_value<float>(scales, input_width, output_width);

        stream.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(gdim * bdim, bdim),
              [=](sycl::nd_item<1> item) {
                upsample_nearest1d_out_frame_kernel<scalar_t, nn_compute_source_index_fn>(
                    item, idata, nbatch, channels, input_width, output_width, odata, scale_factor);
              });
        });
      });
}

template<nn_compute_source_index_fn_t nn_bw_compute_source_index_fn>
static void upsample_nearest1d_backward_out_opencl_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      "upsample_nearest1d_backward_out_opencl_template",
      {grad_output_arg, grad_input_arg});

  int output_width = output_size[0];
  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_width = input_size[2];

  Tensor grad_output = grad_output_.contiguous();

  if (grad_input.numel() == 0) {
    return;
  }

  unsigned int n = grad_input.numel() / nbatch;
  unsigned int bdim = std::min<unsigned int>(MAX_THREADS, MAX_THREADS);
  unsigned int gdim = ceil_div(n, bdim);
  TORCH_CHECK(grad_input.numel() <= std::numeric_limits<int32_t>::max(),
    "upsample_nearest1d_backward only supports input tensors with less than INT_MAX elements, but got ", grad_input.sizes());
  TORCH_CHECK(grad_output.numel() <= std::numeric_limits<int32_t>::max(),
        "upsample_nearest1d_backward only supports output tensors with less than INT_MAX elements, but got ", grad_output.sizes());

  auto stream = at::sycl::getCurrentSYCLStream();
  AT_DISPATCH_FLOATING_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Byte, grad_output.scalar_type(), "upsample_nearest1d_backward_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.mutable_data_ptr<scalar_t>();
        auto odata = grad_output.const_data_ptr<scalar_t>();

        const float scale_factor = compute_scales_value_backwards<float>(scales, output_width, input_width);

        stream.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(gdim * bdim, bdim),
              [=](sycl::nd_item<1> item) {
                upsample_nearest1d_backward_out_frame_kernel<scalar_t, accscalar_t, nn_bw_compute_source_index_fn>(
                    item, odata, nbatch, channels, output_width, input_width, idata, scale_factor);
              });
        });
      });
}

} // namespace

TORCH_IMPL_FUNC(upsample_nearest1d_out_opencl) (
    const Tensor& input,
    IntArrayRef output_size,
    std::optional<double> scales,
    const Tensor& output
) {
  upsample_nearest1d_out_opencl_template<nearest_neighbor_compute_source_index>(
      output, input, output_size, scales);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_out_opencl) (
    const Tensor& input,
    IntArrayRef output_size,
    std::optional<double> scales,
    const Tensor& output
) {
  upsample_nearest1d_out_opencl_template<nearest_neighbor_exact_compute_source_index>(output, input, output_size, scales);
}

TORCH_IMPL_FUNC(upsample_nearest1d_backward_out_opencl) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales,
    const Tensor& grad_input
) {
  upsample_nearest1d_backward_out_opencl_template<nearest_neighbor_bw_compute_source_index>(
      grad_input, grad_output, output_size, input_size, scales);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_backward_out_opencl) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales,
    const Tensor& grad_input
) {
  upsample_nearest1d_backward_out_opencl_template<nearest_neighbor_exact_bw_compute_source_index>(
      grad_input, grad_output, output_size, input_size, scales);
}

} // namespace at::native
