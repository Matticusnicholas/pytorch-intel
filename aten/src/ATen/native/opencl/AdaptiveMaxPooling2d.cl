// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/AdaptiveMaxPooling2d.cu
// SYCL TODO: needs_review - complex kernel requires manual optimization for Intel GPU
// Contains raw CUDA kernels with <<<>>> launch, gpuAtomicAddNoReturn patterns.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/sycl/Atomic.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/sycl/NumericLimits.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_max_pool2d_backward_native.h>
#include <ATen/ops/adaptive_max_pool2d_native.h>
#include <ATen/ops/empty.h>
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace at::native {

namespace {

inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (a / b) * c + ((a % b) * c) / b;
}

inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return 1 + ((a + 1) * c - 1) / b;
}

template <typename T>
void adaptivemaxpool_sycl(
    sycl::queue& queue,
    const T *input, T *output, int64_t *indices,
    int isizeH, int isizeW,
    int osizeH, int osizeW,
    int64_t istrideD, int64_t istrideH, int64_t istrideW,
    int num_planes, int blocksH) {

  sycl::range<3> threads(32, 8, 1);
  sycl::range<3> blocks(1, blocksH, num_planes);

  queue.parallel_for(
      sycl::nd_range<3>(blocks * threads, threads),
      [=](sycl::nd_item<3> item) {
        int o_plane = item.get_group(2);
        int i_plane = o_plane;

        int ostartW = item.get_local_id(0);
        int oendW = osizeW;
        const int ostepW = item.get_local_range(0);

        int ostartH = item.get_local_range(1) * item.get_group(1) + item.get_local_id(1);
        int oendH = osizeH;
        const int ostepH = item.get_local_range(1) * item.get_group_range(1);

        const T* input_p = input + i_plane * istrideD;
        T* output_p = output + o_plane * osizeH * osizeW;
        int64_t* indices_p = indices + o_plane * osizeH * osizeW;

        for (int oh = ostartH; oh < oendH; oh += ostepH) {
          int istartH = start_index(oh, osizeH, isizeH);
          int iendH = end_index(oh, osizeH, isizeH);
          int kH = iendH - istartH;

          for (int ow = ostartW; ow < oendW; ow += ostepW) {
            int istartW = start_index(ow, osizeW, isizeW);
            int iendW = end_index(ow, osizeW, isizeW);
            int kW = iendW - istartW;

            const T *ptr_input = input_p + istartH * istrideH + istartW * istrideW;
            int argmax = istartH * isizeW + istartW;
            T max = at::numeric_limits<T>::lower_bound();

            for (int ih = 0; ih < kH; ih++) {
              for (int iw = 0; iw < kW; iw++) {
                T val = ptr_input[iw * istrideW];
                if ((val > max) || at::_isnan(val)) {
                  max = val;
                  argmax = (ih + istartH) * isizeW + iw + istartW;
                }
              }
              ptr_input += istrideH;
            }
            output_p[oh * osizeW + ow] = max;
            indices_p[oh * osizeW + ow] = argmax;
          }
        }
      });
}

template <typename T>
void atomicadaptivemaxgradinput_sycl(
    sycl::queue& queue,
    T *gradInput, const T *gradOutput, const int64_t *indices,
    int isizeH, int isizeW, int osizeH, int osizeW,
    int num_planes, int blocksH) {

  sycl::range<3> threads(32, 8, 1);
  sycl::range<3> blocks(1, blocksH, num_planes);

  queue.parallel_for(
      sycl::nd_range<3>(blocks * threads, threads),
      [=](sycl::nd_item<3> item) {
        int o_plane = item.get_group(2);
        int i_plane = o_plane;

        int ostartW = item.get_local_id(0);
        int oendW = osizeW;
        int ostepW = item.get_local_range(0);

        int ostartH = item.get_local_range(1) * item.get_group(1) + item.get_local_id(1);
        int oendH = osizeH;
        int ostepH = item.get_local_range(1) * item.get_group_range(1);

        const T* gradOutput_p = gradOutput + o_plane * osizeH * osizeW;
        T* gradInput_p = gradInput + i_plane * isizeH * isizeW;
        const int64_t* indices_p = indices + o_plane * osizeH * osizeW;

        for (int oh = ostartH; oh < oendH; oh += ostepH) {
          for (int ow = ostartW; ow < oendW; ow += ostepW) {
            T z = gradOutput_p[oh * osizeW + ow];
            int argmax = indices_p[oh * osizeW + ow];
            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_val(gradInput_p[argmax]);
            atomic_val += z;
          }
        }
      });
}

} // namespace

TORCH_IMPL_FUNC(adaptive_max_pool2d_out_opencl)
(const Tensor& input, IntArrayRef output_size,
 const Tensor& output, const Tensor& indices) {
  TensorArg output_arg{output, "output", 1};
  TensorArg indices_arg{indices, "indices", 2};
  TensorArg input_arg{input, "input", 3};
  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});
  if (input.numel() == 0) return;

  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];
  auto& queue = at::sycl::getCurrentSYCLQueue();

  const at::Tensor output_c = output.is_contiguous() ? output : at::empty(output.sizes(), output.options());
  const at::Tensor indices_c = indices.is_contiguous() ? indices : at::empty(indices.sizes(), indices.options());

  if (input.ndimension() == 3) {
    int64_t sizeD = input.size(0);
    int64_t isizeH = input.size(1);
    int64_t isizeW = input.size(2);
    int64_t istrideD = input.stride(0);
    int64_t istrideH = input.stride(1);
    int64_t istrideW = input.stride(2);

    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "adaptive_max_pool2d_opencl", [&] {
      int blocksH = std::max((int)(16L / sizeD), 1);
      adaptivemaxpool_sycl<scalar_t>(queue,
          input.const_data_ptr<scalar_t>(), output_c.mutable_data_ptr<scalar_t>(),
          indices_c.mutable_data_ptr<int64_t>(),
          isizeH, isizeW, osizeH, osizeW,
          istrideD, istrideH, istrideW, sizeD, blocksH);
    });
  } else {
    Tensor input_ = input.contiguous();
    int64_t sizeB = input_.size(0);
    int64_t sizeD = input_.size(1);
    int64_t isizeH = input_.size(2);
    int64_t isizeW = input_.size(3);
    int64_t istrideD = isizeH * isizeW;
    int64_t istrideH = input_.stride(2);
    int64_t istrideW = input_.stride(3);

    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input_.scalar_type(), "adaptive_max_pool2d_opencl", [&] {
      int blocksH = std::max((int)(16L / sizeD), 1);
      adaptivemaxpool_sycl<scalar_t>(queue,
          input_.const_data_ptr<scalar_t>(), output_c.mutable_data_ptr<scalar_t>(),
          indices_c.mutable_data_ptr<int64_t>(),
          isizeH, isizeW, osizeH, osizeW,
          istrideD, istrideH, istrideW, sizeB * sizeD, blocksH);
    });
  }

  if (!output.is_contiguous()) output.copy_(output_c);
  if (!indices.is_contiguous()) indices.copy_(indices_c);
}

TORCH_IMPL_FUNC(adaptive_max_pool2d_backward_out_opencl)
(const Tensor& gradOutput, const Tensor& input, const Tensor& indices, const Tensor& gradInput) {
  globalContext().alertNotDeterministic("adaptive_max_pool2d_backward_opencl");

  TensorArg grad_input_arg{gradInput, "gradInput", 1};
  TensorArg grad_output_arg{gradOutput, "gradOutput", 2};
  TensorArg input_arg{input, "input", 3};
  TensorArg indices_arg{indices, "indices", 4};
  checkAllSameGPU(__func__, {grad_input_arg, grad_output_arg, input_arg, indices_arg});

  if (gradOutput.numel() == 0) return;

  const at::Tensor gradOutput_ = gradOutput.contiguous();
  const at::Tensor indices_ = indices.contiguous();
  const at::Tensor gradInput_c = gradInput.is_contiguous() ? gradInput : at::empty(gradInput.sizes(), gradInput.options());
  gradInput_c.zero_();

  auto& queue = at::sycl::getCurrentSYCLQueue();

  if (input.ndimension() == 3) {
    int64_t sizeD = input.size(0);
    int64_t isizeH = input.size(1);
    int64_t isizeW = input.size(2);
    int64_t osizeH = gradOutput_.size(1);
    int64_t osizeW = gradOutput_.size(2);

    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "adaptive_max_pool2d_backward_opencl", [&] {
      int blocksH = std::max((int)(16L / sizeD), 1);
      atomicadaptivemaxgradinput_sycl<scalar_t>(queue,
          gradInput_c.mutable_data_ptr<scalar_t>(), gradOutput_.const_data_ptr<scalar_t>(),
          indices_.const_data_ptr<int64_t>(),
          isizeH, isizeW, osizeH, osizeW, sizeD, blocksH);
    });
  } else {
    int64_t sizeB = input.size(0);
    int64_t sizeD = input.size(1);
    int64_t isizeH = input.size(2);
    int64_t isizeW = input.size(3);
    int64_t osizeH = gradOutput_.size(2);
    int64_t osizeW = gradOutput_.size(3);

    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "adaptive_max_pool2d_backward_opencl", [&] {
      int blocksH = std::max((int)(16L / sizeD), 1);
      atomicadaptivemaxgradinput_sycl<scalar_t>(queue,
          gradInput_c.mutable_data_ptr<scalar_t>(), gradOutput_.const_data_ptr<scalar_t>(),
          indices_.const_data_ptr<int64_t>(),
          isizeH, isizeW, osizeH, osizeW, sizeB * sizeD, blocksH);
    });
  }

  if (!gradInput.is_contiguous()) gradInput.copy_(gradInput_c);
}

} // namespace at::native
