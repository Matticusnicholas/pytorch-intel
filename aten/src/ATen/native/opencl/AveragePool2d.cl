// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/AveragePool2d.cu
// SYCL TODO: needs_review - complex kernel requires manual optimization for Intel GPU
// Contains raw CUDA kernels with CUDA_KERNEL_LOOP, <<<>>> launch, device properties.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Pool.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/sycl/detail/TensorInfo.h>
#include <ATen/sycl/detail/IndexUtils.h>
#include <ATen/sycl/detail/KernelUtils.h>
#include <c10/macros/Macros.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/avg_pool2d_native.h>
#include <ATen/ops/avg_pool2d_backward_native.h>
#endif

namespace at::native {
namespace {

inline int sycl_min(int a, int b) { return a <= b ? a : b; }
inline int sycl_max(int a, int b) { return a >= b ? a : b; }

template <typename scalar_t, typename accscalar_t>
void avg_pool2d_out_sycl_frame(
    sycl::queue& queue,
    const int nthreads,
    const scalar_t* bottom_data, const int64_t channels,
    const int64_t height, const int64_t width,
    const int64_t pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    scalar_t* top_data, const int divisor_override,
    const bool count_include_pad, const bool use_divisor) {

  const uint32_t num_threads = 256;
  const uint32_t num_blocks = (nthreads + num_threads - 1) / num_threads;

  queue.parallel_for(
      sycl::nd_range<1>(num_blocks * num_threads, num_threads),
      [=](sycl::nd_item<1> item) {
        for (int index = item.get_global_id(0); index < nthreads;
             index += item.get_global_range(0)) {
          const int pw = index % pooled_width;
          const int ph = (index / pooled_width) % pooled_height;
          const int c = (index / pooled_width / pooled_height) % channels;
          const int n = index / pooled_width / pooled_height / channels;
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = sycl_min(hstart + kernel_h, (int)height + pad_h);
          int wend = sycl_min(wstart + kernel_w, (int)width + pad_w);
          const int pool_size = (hend - hstart) * (wend - wstart);
          hstart = sycl_max(hstart, 0);
          wstart = sycl_max(wstart, 0);
          hend = sycl_min(hend, (int)height);
          wend = sycl_min(wend, (int)width);

          if (hstart >= hend || wstart >= wend) {
            top_data[index] = scalar_t(0);
            continue;
          }

          accscalar_t aveval = accscalar_t(0);
          const scalar_t* bottom_slice = bottom_data + (n * channels + c) * height * width;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              aveval += bottom_slice[h * width + w];
            }
          }
          int divide_factor;
          if (use_divisor) {
            divide_factor = divisor_override;
          } else {
            divide_factor = count_include_pad ? pool_size : (hend - hstart) * (wend - wstart);
          }
          top_data[index] = static_cast<scalar_t>(aveval / divide_factor);
        }
      });
}

} // anonymous namespace

TORCH_IMPL_FUNC(avg_pool2d_out_opencl)
(const Tensor& input_,
 int64_t kH_, int64_t kW_, int64_t dH_, int64_t dW_,
 int64_t padH_, int64_t padW_,
 bool ceil_mode, bool count_include_pad,
 std::optional<int64_t> divisor_override,
 const Tensor& output) {
  TensorArg output_arg{ output, "output", 1 };
  TensorArg input_arg{ input_, "input_", 2 };
  checkAllSameGPU("avg_pool2d_out_opencl", {output_arg, input_arg});

  const int kH = safe_downcast<int, int64_t>(kH_);
  const int kW = safe_downcast<int, int64_t>(kW_);
  const int dH = safe_downcast<int, int64_t>(dH_);
  const int dW = safe_downcast<int, int64_t>(dW_);
  const int padH = safe_downcast<int, int64_t>(padH_);
  const int padW = safe_downcast<int, int64_t>(padW_);

  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  Tensor input = input_.contiguous();

  const auto count = safe_downcast<int32_t, int64_t>(output.numel());
  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value = use_divisor ? divisor_override.value() : 0;

  if (count != 0) {
    auto& queue = at::sycl::getCurrentSYCLQueue();
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
      "avg_pool2d_out_opencl_frame", [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        avg_pool2d_out_sycl_frame<scalar_t, accscalar_t>(
            queue, count,
            input.const_data_ptr<scalar_t>(), nInputPlane,
            inputHeight, inputWidth, outputHeight, outputWidth,
            kH, kW, dH, dW, padH, padW,
            output.mutable_data_ptr<scalar_t>(),
            divisor_override_value, count_include_pad, use_divisor);
      });
  }
}

TORCH_IMPL_FUNC(avg_pool2d_backward_out_opencl)(
  const Tensor& gradOutput_, const Tensor& input_,
  IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding,
  bool ceil_mode, bool count_include_pad,
  std::optional<int64_t> divisor_override,
  const Tensor& gradInput) {
  // SYCL TODO: needs_review - backward kernel with CUDA_KERNEL_LOOP_TYPE needs SYCL port
  TORCH_CHECK(false, "avg_pool2d_backward_out_opencl: backward kernel pending full SYCL port");
}

} // namespace at::native
