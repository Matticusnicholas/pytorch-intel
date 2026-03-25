// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/AdaptiveAveragePooling.cu
// SYCL TODO: needs_review - complex kernel requires manual optimization for Intel GPU
// This file contains complex raw CUDA kernels with shared memory, <<<>>> launch syntax,
// and __syncthreads(). Full SYCL port requires manual kernel rewrite.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/sycl/Atomic.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/TensorUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/Utils.h>
#include <c10/util/Exception.h>
#include <ATen/native/sycl/LaunchUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_adaptive_avg_pool2d_backward_native.h>
#include <ATen/ops/_adaptive_avg_pool2d_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <ATen/native/AdaptivePooling.h>

#include <algorithm>
#include <cfloat>
#include <cmath>

#define START_IND(a,b,c) ((int64_t)((a / b) * c + ((a % b) * c) / b))
#define END_IND(a,b,c) (1 + ((int64_t)(a + 1) * c - 1) / b)

#define START_IND_INT(a,b,c) ((a * c) / b)
#define END_IND_INT(a,b,c) (((a + 1) * c + b - 1) / b)

#define SYCL_MAX_THREADS 1024
#define BLOCK_STRIDE 2

namespace at::native {

namespace {

  // SYCL kernel: adaptive average pool forward (NCHW)
  template <typename scalar_t>
  void adaptive_average_pool_sycl(
      sycl::queue& queue,
      const scalar_t *input, scalar_t *output,
      int isizeH, int isizeW,
      int osizeH, int osizeW,
      int64_t istrideD, int64_t istrideH, int64_t istrideW,
      int num_planes, int blocksH) {
    using opmath_t = at::opmath_type<scalar_t>;

    sycl::range<3> threads(32, 8, 1);
    sycl::range<3> blocks(1, blocksH, num_planes);

    queue.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item) {
          int o_plane = item.get_group(2);
          int i_plane = o_plane;

          const scalar_t* input_p = input + i_plane * istrideD;
          scalar_t* output_p = output + o_plane * osizeH * osizeW;

          int ostartH = item.get_local_range(1) * item.get_group(1) + item.get_local_id(1);
          int oendH = osizeH;
          const int ostepH = item.get_local_range(1) * item.get_group_range(1);

          int ostartW = item.get_local_id(0);
          int oendW = osizeW;
          const int ostepW = item.get_local_range(0);

          for (int oh = ostartH; oh < oendH; oh += ostepH) {
            int istartH = START_IND(oh, osizeH, isizeH);
            int iendH   = END_IND(oh, osizeH, isizeH);
            int kH = iendH - istartH;

            for (int ow = ostartW; ow < oendW; ow += ostepW) {
              int istartW = START_IND(ow, osizeW, isizeW);
              int iendW   = END_IND(ow, osizeW, isizeW);
              int kW = iendW - istartW;

              const scalar_t *ptr_input = input_p + istartH * istrideH + istartW * istrideW;
              opmath_t sum = static_cast<opmath_t>(0);
              for (int ih = 0; ih < kH; ++ih) {
                for (int iw = 0; iw < kW; ++iw) {
                  sum += ptr_input[iw * istrideW];
                }
                ptr_input += istrideH;
              }
              output_p[oh * osizeW + ow] = sum / kH / kW;
            }
          }
        });
  }

  // SYCL kernel: adaptive average pool backward (gradinput)
  template <typename T>
  void adaptive_average_gradinput_sycl(
      sycl::queue& queue,
      T *gradInput, const T *gradOutput,
      int isizeH, int isizeW, int osizeH, int osizeW,
      int num_planes, int blocksH) {

    sycl::range<3> threads(32, 8, 1);
    sycl::range<3> blocks(1, blocksH, num_planes);

    queue.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item) {
          int i_plane = item.get_group(2);
          int o_plane = i_plane;

          const T* gradOutput_p = gradOutput + o_plane * osizeH * osizeW;
          T* gradInput_p = gradInput + i_plane * isizeH * isizeW;

          int istartH = item.get_local_range(1) * item.get_group(1) + item.get_local_id(1);
          int iendH = isizeH;
          int istepH = item.get_local_range(1) * item.get_group_range(1);

          int istartW = item.get_local_id(0);
          int iendW = isizeW;
          int istepW = item.get_local_range(0);

          for (int ih = istartH; ih < iendH; ih += istepH) {
            int ostartH = START_IND(ih, isizeH, osizeH);
            int oendH   = END_IND(ih, isizeH, osizeH);

            for (int iw = istartW; iw < iendW; iw += istepW) {
              int ostartW = START_IND(iw, isizeW, osizeW);
              int oendW   = END_IND(iw, isizeW, osizeW);

              T *ptr_gradInput = gradInput_p + ih * isizeW + iw;

              for (int oh = ostartH; oh < oendH; ++oh) {
                int kH = START_IND(oh, osizeH, isizeH) - END_IND(oh, osizeH, isizeH);
                for (int ow = ostartW; ow < oendW; ++ow) {
                  int kW = START_IND(ow, osizeW, isizeW) - END_IND(ow, osizeW, isizeW);
                  T grad_delta = gradOutput_p[ow + oh * osizeW] / kH / kW;
                  *ptr_gradInput += grad_delta;
                }
              }
            }
          }
        });
  }

  // SYCL kernel: atomic adaptive average gradinput
  template <typename T>
  void atomic_adaptive_average_gradinput_sycl(
      sycl::queue& queue,
      T *gradInput, const T *gradOutput,
      int isizeH, int isizeW, int osizeH, int osizeW,
      int num_planes, int blocksH) {

    sycl::range<3> threads(32, 8, 1);
    sycl::range<3> blocks(1, blocksH, num_planes);

    queue.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item) {
          int o_plane = item.get_group(2);
          int i_plane = o_plane;

          const T* gradOutput_p = gradOutput + o_plane * osizeW * osizeH;
          T* gradInput_p = gradInput + i_plane * isizeW * isizeH;

          int ostartH = item.get_local_range(1) * item.get_group(1) + item.get_local_id(1);
          int oendH = osizeH;
          int ostepH = item.get_local_range(1) * item.get_group_range(1);

          int ostartW = item.get_local_id(0);
          int oendW = osizeW;
          int ostepW = item.get_local_range(0);

          for (int oh = ostartH; oh < oendH; oh += ostepH) {
            int istartH = START_IND(oh, osizeH, isizeH);
            int iendH   = END_IND(oh, osizeH, isizeH);
            int kH = iendH - istartH;

            for (int ow = ostartW; ow < oendW; ow += ostepW) {
              int istartW = START_IND(ow, osizeW, isizeW);
              int iendW   = END_IND(ow, osizeW, isizeW);
              int kW = iendW - istartW;

              T *ptr_gradInput = gradInput_p + istartH * isizeW + istartW;
              T grad_delta = gradOutput_p[oh * osizeW + ow] / kW / kH;

              for (int ih = 0; ih < kH; ++ih) {
                for (int iw = 0; iw < kW; ++iw) {
                  // SYCL: use sycl::atomic_ref for atomic add
                  sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space>
                      atomic_val(ptr_gradInput[iw]);
                  atomic_val += grad_delta;
                }
                ptr_gradInput += isizeW;
              }
            }
          }
        });
  }

  // SYCL TODO: needs_review - NHWC kernels with shared memory need manual porting
  // The NHWC adaptive_average_pool_nhwc and adaptive_average_gradinput_nhwc kernels
  // use extern __shared__ memory and __syncthreads(). These require careful SYCL
  // translation using sycl::local_accessor and item.barrier().
  // For now, the NCHW path is fully translated above.

  void adaptive_avg_pool2d_out_opencl_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size)
  {
    TensorArg input_arg{ input, "input", 1 },
              output_arg{ output, "output", 2 };
    checkAllSameGPU(__func__, {input_arg, output_arg});

    TORCH_CHECK(output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
    // SYCL TODO: needs_review - full implementation with NHWC path needs manual optimization
    TORCH_CHECK(false, "adaptive_avg_pool2d_opencl: full implementation pending SYCL port of NHWC kernels");
  }

  void adaptive_avg_pool2d_backward_out_opencl_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input)
  {
    // SYCL TODO: needs_review - full implementation pending
    TORCH_CHECK(false, "adaptive_avg_pool2d_backward_opencl: full implementation pending SYCL port");
  }

} // namespace

Tensor& adaptive_avg_pool2d_out_opencl(const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  adaptive_avg_pool2d_out_opencl_template(output, input, output_size);
  return output;
}

Tensor adaptive_avg_pool2d_opencl(const Tensor& input, IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  adaptive_avg_pool2d_out_opencl_template(output, input, output_size);
  return output;
}

Tensor& adaptive_avg_pool2d_backward_out_opencl(const Tensor& gradOutput_,
    const Tensor& input,
    Tensor& gradInput) {
  adaptive_avg_pool2d_backward_out_opencl_template(gradInput, gradOutput_, input);
  return gradInput;
}

Tensor adaptive_avg_pool2d_backward_opencl(const Tensor& gradOutput_, const Tensor& input) {
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  adaptive_avg_pool2d_backward_out_opencl_template(gradInput, gradOutput_, input);
  return gradInput;
}

} // namespace at::native
