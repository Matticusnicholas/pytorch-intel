// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/SortUtils.cuh
#pragma once
#include <c10/macros/Macros.h>
#include <ATen/native/sycl/SortingCommon.h>
#include <ATen/native/StridedRandomAccessor.h>

// SYCL: cub::WarpMergeSort is replaced by oneDPL or manual implementations
// oneDPL: #include <oneapi/dpl/algorithm>

namespace at::native {

template <typename T>
inline void swapVars(T& t1, T& t2) {
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}

template <typename Comparator, typename K, typename V>
inline void bitonicSwap(K& kA, V& vA, bool& validA,
                                   K& kB, V& vB, bool& validB,
                                   bool dir,
                                   const Comparator& comp) {
  bool swap = (comp(kA, kB) && validA) || !validB;
  if (swap == dir) {
    swapVars(kA, kB);
    swapVars(vA, vB);
    swapVars(validA, validB);
  }
}

// SYCL: Bitonic sort within a work-group
// Uses sycl::nd_item::get_local_id(0) instead of threadIdx.x
// Uses sycl::group_barrier instead of __syncthreads()
template <int Power2SortSize, typename IndexType, typename Comparator,
          typename K, typename V>
inline void bitonicSort(K *keys,
                                   V *values,
                                   bool *valid,
                                   const Comparator& comp,
                                   sycl::nd_item<1> item) {
#pragma unroll
  for (unsigned int size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((item.get_local_id(0) & (size / 2)) != 0);

#pragma unroll
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {
      sycl::group_barrier(item.get_group());

      unsigned int pos = 2 * item.get_local_id(0) - (item.get_local_id(0) & (stride - 1));
      bitonicSwap<Comparator, K, V>(
        keys[pos], values[pos], valid[pos],
        keys[pos + stride], values[pos + stride], valid[pos + stride],
        flag, comp);
    }
  }

#pragma unroll
  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {
    sycl::group_barrier(item.get_group());

    unsigned int pos = 2 * item.get_local_id(0) - (item.get_local_id(0) & (stride - 1));
    bitonicSwap<Comparator, K, V>(
      keys[pos], values[pos], valid[pos],
      keys[pos + stride], values[pos + stride], valid[pos + stride],
      false, comp);
  }

  sycl::group_barrier(item.get_group());
}

} // namespace at::native
