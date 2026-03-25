// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/AdaptiveMaxPooling3d.cu
// SYCL TODO: needs_review - complex kernel requires manual optimization for Intel GPU
// Contains raw CUDA kernels with <<<>>> launch, dim3, atomicAdd, loop patterns.
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
#include <ATen/ops/adaptive_max_pool3d_backward_native.h>
#include <ATen/ops/adaptive_max_pool3d_native.h>
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

// SYCL kernel for 3D adaptive max pooling forward
template <typename T>
void adaptivemaxpool3d_sycl(
    sycl::queue& queue,
    const T *input, T *output, int64_t *indices,
    int64_t totalZ,
    int isizeT, int isizeH, int isizeW,
    int osizeT, int osizeH, int osizeW,
    int64_t istrideD, int64_t istrideT, int64_t istrideH, int64_t istrideW) {

  int64_t offsetZ = 0;
  int blocksH = std::max((int)(16L / totalZ), 1);

  while (totalZ > 0) {
    int64_t curZ = totalZ > 65535 ? 65535 : totalZ;
    sycl::range<2> threads(8, 32);
    sycl::range<2> blocks(blocksH, curZ);
    int64_t capturedOffset = offsetZ;

    queue.parallel_for(
        sycl::nd_range<2>(blocks * threads, threads),
        [=](sycl::nd_item<2> item) {
          int ostartH = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
          int oendH = osizeH;
          int ostepH = item.get_group_range(0) * item.get_local_range(0);
          int ostartW = item.get_local_id(1);
          int oendW = osizeW;
          int ostepW = item.get_local_range(1);

          int64_t o_plane = item.get_group(1) + capturedOffset;
          int ot = o_plane % osizeT;
          int d = o_plane / osizeT;

          int istartT = start_index(ot, osizeT, isizeT);
          int iendT = end_index(ot, osizeT, isizeT);
          int kT = iendT - istartT;

          const T *input_dt = input + d * istrideD + istartT * istrideT;
          T *output_dt = output + o_plane * osizeH * osizeW;
          int64_t *indices_dt = indices + o_plane * osizeH * osizeW;

          for (int oh = ostartH; oh < oendH; oh += ostepH) {
            int istartH_l = start_index(oh, osizeH, isizeH);
            int iendH_l = end_index(oh, osizeH, isizeH);
            int kH = iendH_l - istartH_l;

            for (int ow = ostartW; ow < oendW; ow += ostepW) {
              int istartW_l = start_index(ow, osizeW, isizeW);
              int iendW_l = end_index(ow, osizeW, isizeW);
              int kW = iendW_l - istartW_l;

              const T *ptr_input = input_dt + istartH_l * istrideH + istartW_l * istrideW;
              int64_t argmax = istartT * isizeH * isizeW + istartH_l * isizeW + istartW_l;
              T max = at::numeric_limits<T>::lower_bound();

              for (int it = 0; it < kT; ++it) {
                for (int ih = 0; ih < kH; ++ih) {
                  for (int iw = 0; iw < kW; ++iw) {
                    T val = ptr_input[ih * istrideH + iw * istrideW];
                    if ((val > max) || at::_isnan(val)) {
                      max = val;
                      argmax = (it + istartT) * isizeH * isizeW + (ih + istartH_l) * isizeW + iw + istartW_l;
                    }
                  }
                }
                ptr_input += istrideT;
              }
              output_dt[oh * osizeW + ow] = max;
              indices_dt[oh * osizeW + ow] = argmax;
            }
          }
        });

    totalZ -= 65535;
    offsetZ += 65535;
  }
}

} // namespace

TORCH_IMPL_FUNC(adaptive_max_pool3d_out_opencl)
(const Tensor& input, IntArrayRef output_size,
 const Tensor& output, const Tensor& indices) {
  TensorArg output_arg{output, "output", 1};
  TensorArg indices_arg{indices, "indices", 2};
  TensorArg input_arg{input, "input", 3};
  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});
  if (input.numel() == 0 || output.numel() == 0) return;

  int64_t osizeT = output_size[0];
  int64_t osizeH = output_size[1];
  int64_t osizeW = output_size[2];

  int64_t sizeD, isizeT, isizeH, isizeW;
  int64_t istrideD, istrideT, istrideH, istrideW;
  int64_t totalZ;

  const Tensor& input_ = input.ndimension() == 4 ? input : input.contiguous();

  if (input_.ndimension() == 4) {
    sizeD = input_.size(0);
    isizeT = input_.size(1); isizeH = input_.size(2); isizeW = input_.size(3);
    istrideD = input_.stride(0); istrideT = input_.stride(1);
    istrideH = input_.stride(2); istrideW = input_.stride(3);
    totalZ = sizeD * osizeT;
  } else {
    int64_t sizeB = input_.size(0);
    sizeD = input_.size(1);
    isizeT = input_.size(2); isizeH = input_.size(3); isizeW = input_.size(4);
    istrideD = isizeT * isizeH * isizeW;
    istrideT = input_.stride(2); istrideH = input_.stride(3); istrideW = input_.stride(4);
    totalZ = sizeB * sizeD * osizeT;
  }

  auto& queue = at::sycl::getCurrentSYCLQueue();
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input_.scalar_type(), "adaptive_max_pool3d_opencl", [&] {
    adaptivemaxpool3d_sycl<scalar_t>(queue,
        input_.const_data_ptr<scalar_t>(), output.mutable_data_ptr<scalar_t>(),
        indices.mutable_data_ptr<int64_t>(), totalZ,
        isizeT, isizeH, isizeW, osizeT, osizeH, osizeW,
        istrideD, istrideT, istrideH, istrideW);
  });
}

TORCH_IMPL_FUNC(adaptive_max_pool3d_backward_out_opencl)
(const Tensor& gradOutput, const Tensor& input, const Tensor& indices, const Tensor& gradInput) {
  // SYCL TODO: needs_review - backward with atomicAdd needs sycl::atomic_ref port
  TensorArg grad_input_arg{gradInput, "gradInput", 1};
  TensorArg grad_output_arg{gradOutput, "gradOutput", 2};
  TensorArg input_arg{input, "input", 3};
  TensorArg indices_arg{indices, "indices", 4};
  checkAllSameGPU(__func__, {grad_input_arg, grad_output_arg, input_arg, indices_arg});
  if (gradOutput.numel() == 0) return;

  gradInput.zero_();
  // SYCL TODO: needs_review - implement backward kernel with sycl::atomic_ref
  TORCH_CHECK(false, "adaptive_max_pool3d_backward_opencl: backward kernel pending full SYCL port");
}

} // namespace at::native
