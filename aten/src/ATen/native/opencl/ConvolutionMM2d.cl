// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ConvolutionMM2d.cu
// SYCL TODO: needs_review - complex kernel requires manual optimization for Intel GPU
// Uses CUDABlas (gemm), im2col.cuh which need oneMKL/oneDNN equivalents for Intel GPU.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/div_rtn.h>
#include <ATen/sycl/SYCLBlas.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/sycl/im2col.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_slow_conv2d_forward_native.h>
#include <ATen/ops/_slow_conv2d_backward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sum.h>
#endif

namespace at::native {
namespace {

void slow_conv2d_shape_check(
    const Tensor& input, const Tensor& grad_output,
    const Tensor& weight, const Tensor& bias,
    int64_t kH, int64_t kW, int64_t dH, int64_t dW,
    int64_t padH, int64_t padW, bool weight_nullable) {
  TORCH_CHECK(kW > 0 && kH > 0, "kernel size should be greater than zero, but got kH: ", kH, " kW: ", kW);
  TORCH_CHECK(dW > 0 && dH > 0, "stride should be greater than zero, but got dH: ", dH, " dW: ", dW);
  TORCH_CHECK(weight_nullable || weight.defined(), "weight tensor is expected to be non-nullable");
  TORCH_CHECK(!weight.defined() || ((weight.numel() > 0) && (weight.dim() == 2)),
      "non-empty 2D weight tensor expected, but got: ", weight.sizes());
  TORCH_CHECK(!bias.defined() || (bias.dim() == 1 && bias.sizes()[0] == weight.sizes()[0]),
      "Expected bias to have shape [", weight.sizes()[0], "] but got ", bias.sizes());

  const auto in_sizes = input.sizes();
  constexpr int ndim = 4;
  constexpr int dimf = 1;
  constexpr int dimh = 2;
  constexpr int dimw = 3;
  TORCH_CHECK(in_sizes.size() == ndim, "Expected 4D input tensor, but got ", in_sizes);

  const bool valid_empty = c10::multiply_integers(in_sizes.slice(1)) != 0;
  TORCH_CHECK(valid_empty, "non-empty input tensor expected but got: ", in_sizes);

  int64_t inputHeight = in_sizes[dimh];
  int64_t inputWidth = in_sizes[dimw];
  int64_t exactInputHeight = inputHeight + 2 * padH;
  int64_t exactInputWidth = inputWidth + 2 * padW;

  TORCH_CHECK(exactInputHeight >= kH && exactInputWidth >= kW,
      "Calculated padded input size per channel: ", IntArrayRef{exactInputHeight, exactInputWidth},
      ". Kernel size: ", IntArrayRef{kH, kW}, ". Kernel size can't be greater than actual input size");

  auto outputHeight = div_rtn<int64_t>(exactInputHeight - kH, dH) + 1;
  auto outputWidth = div_rtn<int64_t>(exactInputWidth - kW, dW) + 1;

  TORCH_CHECK(outputWidth >= 1 && outputHeight >= 1,
      "Given input size per channel: ", IntArrayRef{inputHeight, inputWidth},
      ". Calculated output size per channel: ", IntArrayRef{outputHeight, outputWidth},
      ". Output size is too small");

  if (weight.defined()) {
    const auto w_sizes = weight.sizes();
    int64_t nInputPlane = w_sizes[1];
    if (w_sizes.size() == 2) {
      nInputPlane /= (kH * kW);
    }
    TORCH_CHECK(in_sizes[dimf] == nInputPlane,
        "Expected input dim ", dimf, " to have size ", nInputPlane, " but got ", in_sizes[dimf]);
  }

  if (grad_output.defined()) {
    const auto gO_sizes = grad_output.sizes();
    TORCH_CHECK(gO_sizes.size() == ndim, "Expected grad_output to have ", ndim, " dimensions but got shape", gO_sizes);
    if (weight.defined()) {
      TORCH_CHECK(gO_sizes[dimf] == weight.sizes()[0],
          "Expected dim ", dimf, " to have size ", weight.sizes()[0], " but got ", gO_sizes[dimf]);
    }
  }
}

} // namespace

// SYCL TODO: needs_review - The full forward and backward implementations use
// CUDABlas gemm calls and im2col/col2im CUDA kernels. These need to be replaced
// with oneMKL gemm and SYCL im2col/col2im kernels for Intel GPU.
// Placeholder implementations that indicate pending port:

Tensor _slow_conv2d_forward_opencl(
    const Tensor& self, const Tensor& weight, IntArrayRef kernel_size,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef stride, IntArrayRef padding) {
  TORCH_CHECK(false, "_slow_conv2d_forward_opencl: pending SYCL port (needs oneMKL gemm + SYCL im2col)");
}

std::tuple<Tensor, Tensor, Tensor> _slow_conv2d_backward_opencl(
    const Tensor& grad_output, const Tensor& self, const Tensor& weight,
    IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding,
    std::array<bool, 3> output_mask) {
  TORCH_CHECK(false, "_slow_conv2d_backward_opencl: pending SYCL port (needs oneMKL gemm + SYCL col2im)");
}

} // namespace at::native
