// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/TensorModeKernel.cuh
#pragma once

#include <ATen/native/sycl/Loops.h>
#include <ATen/native/sycl/SortingCommon.h>
#include <ATen/native/sycl/BlockReduce.h>

namespace at::native {

struct ModeUnsignedBoolPair {
  unsigned int val;
  bool flag;
};

struct ModeUnsignedPair {
  unsigned int val;
  unsigned int index;
};

// SYCL: Inclusive prefix scan using work-group local memory
// Uses sycl::group_barrier instead of __syncthreads()
template <int Power2ScanSize, typename T, class BinaryOp>
void inclusivePrefixScan(T* smem, BinaryOp binop, sycl::nd_item<1> item) {
  // Reduce step ("upsweep")
#pragma unroll
  for (int stride = 1; stride < Power2ScanSize; stride <<= 1) {
    int index = (item.get_local_id(0) + 1) * stride * 2 - 1;
    if (index < Power2ScanSize) {
      smem[index] = binop(smem[index], smem[index - stride]);
    }
    sycl::group_barrier(item.get_group());
  }

  // Post-reduce step ("downsweep")
#pragma unroll
  for (int stride = Power2ScanSize / 4; stride > 0; stride >>= 1) {
    int index = (item.get_local_id(0) + 1) * stride * 2 - 1;
    if ((index + stride) < Power2ScanSize) {
      smem[index + stride] = binop(smem[index + stride], smem[index]);
    }
    sycl::group_barrier(item.get_group());
  }
}

} // namespace at::native
