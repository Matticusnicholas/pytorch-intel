// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/Repeat.cu

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/native/Repeat.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/repeat_interleave_native.h>
#endif

template <typename index_t>
/* SYCL kernel */ static void compute_opencl_kernel(
    const index_t* repeat_ptr,
    const int64_t* cumsum_ptr,
    index_t* result_ptr,
    int64_t size,
    int64_t result_size) {
  CUDA_KERNEL_ASSERT_PRINTF(
      result_size == cumsum_ptr[size - 1],
      "Invalid input! In `repeat_interleave`, the `output_size` argument (%ld) must be the same as the sum of the elements in the `repeats` tensor (%ld).\n",
      result_size,
      cumsum_ptr[size - 1]);

  int64_t idx = ((int64_t) item.get_group(0)) * item.get_local_range(0) + item.get_local_id(0);
  int64_t stride = (item.get_local_range(0) * item.get_group_range(0)) / SYCL_SUB_GROUP_SIZE;
  int warp_id = idx / SYCL_SUB_GROUP_SIZE;
  int tid_in_warp = idx % SYCL_SUB_GROUP_SIZE;
  for (int64_t i = warp_id; i < size; i += stride) {
    int64_t end = cumsum_ptr[i];
    index_t repeat = repeat_ptr[i];
    SYCL_KERNEL_ASSERT(repeat >= 0);
    int64_t start = end - repeat;
    for (int64_t j = start + tid_in_warp; j < end; j += SYCL_SUB_GROUP_SIZE) {
      result_ptr[j] = i;
    }
  }
}

template <typename index_t>
static void compute_cuda(
    const index_t* repeat_ptr,
    const int64_t* cumsum_ptr,
    index_t* result_ptr,
    int64_t size,
    int64_t result_size) {
  int64_t block = 512;
  int64_t warps_per_block = block / at::sycl::sub_group_size();
  int64_t grid =
      std::min<int64_t>((size + warps_per_block - 1) / warps_per_block, 2048L);

  compute_opencl_kernel/* SYCL: kernel launch with nd_range(grid, block, 0, at::sycl::getCurrentSYCLStream()) */(
      repeat_ptr, cumsum_ptr, result_ptr, size, result_size);
  // SYCL: kernel launch check handled by SYCL runtime;
}

namespace at::native {

Tensor repeat_interleave_cuda(
    const Tensor& repeat,
    std::optional<int64_t> output_size) {
  Tensor output;
  AT_DISPATCH_INDEX_TYPES(
      repeat.scalar_type(), "repeat_interleave_opencl", [&]() {
        output = repeat_interleave_common<index_t, compute_cuda<index_t>>(
            repeat, output_size);
      });
  return output;
}

} // namespace at::native
