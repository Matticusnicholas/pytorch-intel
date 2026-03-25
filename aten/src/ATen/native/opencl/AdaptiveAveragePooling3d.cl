// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/AdaptiveAveragePooling3d.cu
// SYCL TODO: needs_review - complex kernel requires manual optimization for Intel GPU
// Contains raw CUDA kernels with <<<>>> launch, dim3, atomicAdd patterns.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/sycl/Atomic.h>
#include <ATen/sycl/SYCLContext.h>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_avg_pool3d_backward_native.h>
#include <ATen/ops/adaptive_avg_pool3d_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

#include <ATen/native/AdaptivePooling.h>

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

// SYCL kernel for 3D adaptive average pooling forward
template <typename scalar_t, typename accscalar_t>
void adaptiveaveragepool_sycl(
    sycl::queue& queue,
    const scalar_t *input, scalar_t *output,
    int isizeT, int isizeH, int isizeW,
    int osizeT, int osizeH, int osizeW,
    int64_t sizeD, int64_t istrideB, int64_t istrideD,
    int64_t istrideT, int64_t istrideH, int64_t istrideW,
    int64_t totalZ) {

  int64_t offsetZ = 0;
  int blocksH = std::max((int)(16L / totalZ), 1);

  while (totalZ > 0) {
    int64_t curZ = totalZ > 65535 ? 65535 : totalZ;
    sycl::range<2> threads(8, 32);
    sycl::range<2> blocks(blocksH, curZ);

    queue.parallel_for(
        sycl::nd_range<2>(blocks * threads, threads),
        [=](sycl::nd_item<2> item) {
          int ostartH = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
          int oendH = osizeH;
          int ostepH = item.get_group_range(0) * item.get_local_range(0);
          int ostartW = item.get_local_id(1);
          int oendW = osizeW;
          int ostepW = item.get_local_range(1);

          int64_t o_plane = item.get_group(1) + offsetZ;
          int ot = o_plane % osizeT;
          int d = o_plane / osizeT;
          int batch_idx = d / sizeD;
          int channel_idx = d % sizeD;

          int istartT = start_index(ot, osizeT, isizeT);
          int iendT = end_index(ot, osizeT, isizeT);
          int kT = iendT - istartT;

          scalar_t *output_dt = output + o_plane * osizeH * osizeW;

          for (int oh = ostartH; oh < oendH; oh += ostepH) {
            int istartH_l = start_index(oh, osizeH, isizeH);
            int iendH_l = end_index(oh, osizeH, isizeH);
            int kH = iendH_l - istartH_l;

            for (int ow = ostartW; ow < oendW; ow += ostepW) {
              int istartW_l = start_index(ow, osizeW, isizeW);
              int iendW_l = end_index(ow, osizeW, isizeW);
              int kW = iendW_l - istartW_l;

              accscalar_t sum = static_cast<accscalar_t>(0);
              for (int it = 0; it < kT; ++it) {
                for (int ih = 0; ih < kH; ++ih) {
                  for (int iw = 0; iw < kW; ++iw) {
                    int64_t input_offset = batch_idx * istrideB + channel_idx * istrideD +
                                           (istartT + it) * istrideT +
                                           (istartH_l + ih) * istrideH + (istartW_l + iw) * istrideW;
                    sum += static_cast<accscalar_t>(input[input_offset]);
                  }
                }
              }
              const accscalar_t divide_factor = static_cast<accscalar_t>(kT * kH * kW);
              output_dt[oh * osizeW + ow] = static_cast<scalar_t>(sum / divide_factor);
            }
          }
        });

    totalZ -= 65535;
    offsetZ += 65535;
  }
}

void adaptive_avg_pool3d_out_opencl_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef& output_size) {
  TensorArg output_arg{output, "output", 1};
  TensorArg input_arg{input_, "input_", 2};
  checkAllSameGPU("adaptive_avg_pool3d_opencl", {output_arg, input_arg});

  for (int64_t i = 1; i < input_.ndimension(); i++) {
    TORCH_CHECK(input_.size(i) > 0,
        "adaptive_avg_pool3d_opencl(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ", input_.sizes(), " with dimension ", i, " being empty");
  }

  TORCH_CHECK((input_.ndimension() == 4 || input_.ndimension() == 5),
      "adaptive_avg_pool3d_opencl(): Expected 4D or 5D tensor, but got ", input_.sizes());

  TORCH_CHECK(output_size.size() == 1 || output_size.size() == 3,
      "adaptive_avg_pool3d: internal error: output_size.size() must be 1 or 3");

  int64_t osizeT = output_size[0];
  int64_t osizeH = output_size[1];
  int64_t osizeW = output_size[2];

  int64_t sizeD, isizeT, isizeH, isizeW;
  int64_t istrideB, istrideD, istrideT, istrideH, istrideW;
  int64_t totalZ;

  const Tensor& input = input_.ndimension() == 4 ? input_ : input_.contiguous();

  if (input.ndimension() == 4) {
    sizeD = input.size(0);
    isizeT = input.size(1); isizeH = input.size(2); isizeW = input.size(3);
    istrideB = 0; istrideD = input.stride(0);
    istrideT = input.stride(1); istrideH = input.stride(2); istrideW = input.stride(3);
    output.resize_({sizeD, osizeT, osizeH, osizeW});
    totalZ = sizeD * osizeT;
  } else {
    int64_t sizeB = input.size(0);
    sizeD = input.size(1);
    isizeT = input.size(2); isizeH = input.size(3); isizeW = input.size(4);
    istrideB = input.stride(0); istrideD = input.stride(1);
    istrideT = input.stride(2); istrideH = input.stride(3); istrideW = input.stride(4);
    output.resize_({sizeB, sizeD, osizeT, osizeH, osizeW});
    totalZ = sizeB * sizeD * osizeT;
  }

  if (output.numel() == 0) return;

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
      input.scalar_type(), "adaptive_avg_pool3d_opencl", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto& queue = at::sycl::getCurrentSYCLQueue();
        adaptiveaveragepool_sycl<scalar_t, accscalar_t>(
            queue,
            input.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
            totalZ, isizeT, isizeH, isizeW,
            osizeT, osizeH, osizeW,
            sizeD, istrideB, istrideD, istrideT, istrideH, istrideW);
      });
}

// SYCL TODO: needs_review - backward kernels with atomicAdd need sycl::atomic_ref
void adaptive_avg_pool3d_backward_out_opencl_template(
    Tensor& gradInput, const Tensor& gradOutput_, const Tensor& input) {
  TORCH_CHECK(false, "adaptive_avg_pool3d_backward_opencl: full backward implementation pending SYCL port");
}

} // namespace

Tensor& adaptive_avg_pool3d_out_opencl(const Tensor& input,
    IntArrayRef output_size, Tensor& output) {
  adaptive_avg_pool3d_out_opencl_template(output, input, output_size);
  return output;
}

Tensor adaptive_avg_pool3d_opencl(const Tensor& input, IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  adaptive_avg_pool3d_out_opencl_template(output, input, output_size);
  return output;
}

Tensor& adaptive_avg_pool3d_backward_out_opencl(const Tensor& gradOutput_,
    const Tensor& input, Tensor& gradInput) {
  globalContext().alertNotDeterministic("adaptive_avg_pool3d_backward_out_opencl");
  adaptive_avg_pool3d_backward_out_opencl_template(gradInput, gradOutput_, input);
  return gradInput;
}

Tensor adaptive_avg_pool3d_backward_opencl(const Tensor& gradOutput_, const Tensor& input) {
  globalContext().alertNotDeterministic("adaptive_avg_pool3d_backward_opencl");
  auto gradInput = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  adaptive_avg_pool3d_backward_out_opencl_template(gradInput, gradOutput_, input);
  return gradInput;
}

} // namespace at::native
