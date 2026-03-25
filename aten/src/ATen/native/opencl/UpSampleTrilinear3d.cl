// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/UpSampleTrilinear3d.cu

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/native/sycl/Atomic.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/sycl/SYCLApplyUtils.h>
#include <ATen/native/sycl/UpSample.h>
#include <ATen/native/sycl/KernelUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/upsample_trilinear3d_native.h>
#include <ATen/ops/upsample_trilinear3d_backward_native.h>
#endif

namespace at::native {
namespace {

inline size_t
idx_3d(const size_t nc,
    const size_t depth,
    const size_t height,
    const size_t width,
    const size_t z,
    const size_t y,
    const size_t x) {
  return ((nc * depth + z) * height + y) * width + x;
}

template <typename scalar_t, typename accscalar_t>
void upsample_trilinear3d_out_frame_kernel(
    sycl::nd_item<1> item,
    const int n,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor64<const scalar_t, 5> idata,
    PackedTensorAccessor64<scalar_t, 5> odata) {
  int index = item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int depth1 = idata.size(2);
  const int height1 = idata.size(3);
  const int width1 = idata.size(4);
  const int depth2 = odata.size(2);
  const int height2 = odata.size(3);
  const int width2 = odata.size(4);

  if (index < n) {
    const int w2 = (index % (height2 * width2)) % width2;
    const int h2 = (index % (height2 * width2)) / width2;
    const int t2 = index / (height2 * width2);

    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int t1 = t2;
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = idata[n][c][t1][h1][w1];
          odata[n][c][t2][h2][w2] = val;
        }
      }
      return;
    }

    const accscalar_t t1r = area_pixel_compute_source_index<accscalar_t>(rdepth, t2, align_corners, false);
    const int t1 = t1r;
    const int t1p = (t1 < depth1 - 1) ? 1 : 0;
    const accscalar_t t1lambda = t1r - t1;
    const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;

    const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(rheight, h2, align_corners, false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(rwidth, w2, align_corners, false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const accscalar_t val = t0lambda *
                (h0lambda *
                     (w0lambda * idata[n][c][t1][h1][w1] +
                      w1lambda * idata[n][c][t1][h1][w1 + w1p]) +
                 h1lambda *
                     (w0lambda * idata[n][c][t1][h1 + h1p][w1] +
                      w1lambda * idata[n][c][t1][h1 + h1p][w1 + w1p])) +
            t1lambda *
                (h0lambda *
                     (w0lambda * idata[n][c][t1 + t1p][h1][w1] +
                      w1lambda * idata[n][c][t1 + t1p][h1][w1 + w1p]) +
                 h1lambda *
                     (w0lambda * idata[n][c][t1 + t1p][h1 + h1p][w1] +
                      w1lambda * idata[n][c][t1 + t1p][h1 + h1p][w1 + w1p]));
        odata[n][c][t2][h2][w2] = static_cast<scalar_t>(val);
      }
    }
  }
}

template <typename scalar_t, typename accscalar_t>
void upsample_trilinear3d_backward_out_frame_kernel(
    sycl::nd_item<1> item,
    const int num_kernels,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    PackedTensorAccessor64<scalar_t, 5> idata,
    const PackedTensorAccessor64<const scalar_t, 5> odata,
    scalar_t* idata_ptr) {
  int index = item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int depth1 = idata.size(2);
  const int height1 = idata.size(3);
  const int width1 = idata.size(4);
  const int depth2 = odata.size(2);
  const int height2 = odata.size(3);
  const int width2 = odata.size(4);

  const size_t i_numel = batchsize * channels * depth1 * height1 * width1;

  if (index < num_kernels) {
    const int w2 = (index % (height2 * width2)) % width2;
    const int h2 = (index % (height2 * width2)) / width2;
    const int t2 = index / (height2 * width2);

    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int t1 = t2;
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = odata[n][c][t1][h1][w1];
          idata[n][c][t2][h2][w2] = val;
        }
      }
      return;
    }

    const accscalar_t t1r = area_pixel_compute_source_index<accscalar_t>(rdepth, t2, align_corners, false);
    const int t1 = t1r;
    const int t1p = (t1 < depth1 - 1) ? 1 : 0;
    const accscalar_t t1lambda = t1r - t1;
    const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;

    const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(rheight, h2, align_corners, false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(rwidth, w2, align_corners, false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const scalar_t d2val = odata[n][c][t2][h2][w2];
        const size_t nc = n * channels + c;
        // SYCL: use sycl::atomic_ref for atomic add operations
        auto do_atomic_add = [&](size_t idx, scalar_t val) {
          sycl::atomic_ref<scalar_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                           sycl::access::address_space::global_space> ref(idata_ptr[idx]);
          ref += val;
        };
        do_atomic_add(idx_3d(nc, depth1, height1, width1, t1, h1, w1),
            static_cast<scalar_t>(t0lambda * h0lambda * w0lambda * d2val));
        do_atomic_add(idx_3d(nc, depth1, height1, width1, t1, h1, w1 + w1p),
            static_cast<scalar_t>(t0lambda * h0lambda * w1lambda * d2val));
        do_atomic_add(idx_3d(nc, depth1, height1, width1, t1, h1 + h1p, w1),
            static_cast<scalar_t>(t0lambda * h1lambda * w0lambda * d2val));
        do_atomic_add(idx_3d(nc, depth1, height1, width1, t1, h1 + h1p, w1 + w1p),
            static_cast<scalar_t>(t0lambda * h1lambda * w1lambda * d2val));
        do_atomic_add(idx_3d(nc, depth1, height1, width1, t1 + t1p, h1, w1),
            static_cast<scalar_t>(t1lambda * h0lambda * w0lambda * d2val));
        do_atomic_add(idx_3d(nc, depth1, height1, width1, t1 + t1p, h1, w1 + w1p),
            static_cast<scalar_t>(t1lambda * h0lambda * w1lambda * d2val));
        do_atomic_add(idx_3d(nc, depth1, height1, width1, t1 + t1p, h1 + h1p, w1),
            static_cast<scalar_t>(t1lambda * h1lambda * w0lambda * d2val));
        do_atomic_add(idx_3d(nc, depth1, height1, width1, t1 + t1p, h1 + h1p, w1 + w1p),
            static_cast<scalar_t>(t1lambda * h1lambda * w1lambda * d2val));
      }
    }
  }
}

static void upsample_trilinear3d_out_opencl_template(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_trilinear3d_out_opencl", {input_arg, output_arg});

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];
  int input_depth = input.size(2);
  int input_height = input.size(3);
  int input_width = input.size(4);

  const int num_kernels = output_depth * output_height * output_width;
  const int num_threads = 512;
  auto stream = at::sycl::getCurrentSYCLStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      input.scalar_type(), "upsample_trilinear3d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor64<const scalar_t, 5>();
        auto odata = output.packed_accessor64<scalar_t, 5>();

        const accscalar_t rdepth = area_pixel_compute_scale<accscalar_t>(input_depth, output_depth, align_corners, scales_d);
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(input_width, output_width, align_corners, scales_w);

        int groups = ceil_div(num_kernels, num_threads);
        stream.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(groups * num_threads, num_threads),
              [=](sycl::nd_item<1> item) {
                upsample_trilinear3d_out_frame_kernel<scalar_t, accscalar_t>(
                    item, num_kernels, rdepth, rheight, rwidth, align_corners, idata, odata);
              });
        });
      });
}

static void upsample_trilinear3d_backward_out_opencl_template(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg grad_input_arg{grad_input_, "grad_input_", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU("upsample_trilinear3d_backward_out_opencl", {grad_output_arg, grad_input_arg});

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];
  int input_depth = input_size[2];
  int input_height = input_size[3];
  int input_width = input_size[4];

  Tensor grad_output = grad_output_.contiguous();
  Tensor grad_input = grad_input_.contiguous();
  grad_input.zero_();

  const int num_kernels = output_depth * output_height * output_width;
  const int num_threads = 256;
  auto stream = at::sycl::getCurrentSYCLStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      grad_output.scalar_type(), "upsample_trilinear3d_backward_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.packed_accessor64<scalar_t, 5>();
        auto odata = grad_output.packed_accessor64<const scalar_t, 5>();
        scalar_t* idata_ptr = grad_input.mutable_data_ptr<scalar_t>();

        const accscalar_t rdepth = area_pixel_compute_scale<accscalar_t>(input_depth, output_depth, align_corners, scales_d);
        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(input_width, output_width, align_corners, scales_w);

        int groups = ceil_div(num_kernels, num_threads);
        stream.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(groups * num_threads, num_threads),
              [=](sycl::nd_item<1> item) {
                upsample_trilinear3d_backward_out_frame_kernel<scalar_t, accscalar_t>(
                    item, num_kernels, rdepth, rheight, rwidth, align_corners,
                    idata, odata, idata_ptr);
              });
        });

        if (!grad_input_.is_contiguous()) {
            grad_input_.copy_(grad_input);
        }
      });
}

} // namespace

TORCH_IMPL_FUNC(upsample_trilinear3d_out_opencl) (
    const Tensor& input, IntArrayRef output_size, bool align_corners,
    std::optional<double> scales_d, std::optional<double> scales_h,
    std::optional<double> scales_w, const Tensor& output) {
  upsample_trilinear3d_out_opencl_template(output, input, output_size, align_corners, scales_d, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_trilinear3d_backward_out_opencl) (
    const Tensor& grad_output, IntArrayRef output_size,
    IntArrayRef input_size, bool align_corners,
    std::optional<double> scales_d, std::optional<double> scales_h,
    std::optional<double> scales_w, const Tensor& grad_input) {
  globalContext().alertNotDeterministic("upsample_trilinear3d_backward_out_opencl");
  upsample_trilinear3d_backward_out_opencl_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
}

} // namespace at::native
