// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/PersistentSoftmax.cuh
#pragma once

#include <cfloat>
#include <limits>
#include <stdint.h>
#include <c10/macros/Macros.h>

namespace {

int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

template<typename T>
struct Add {
  inline T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  inline T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

// SYCL: sub-group size constant
static constexpr int SYCL_WARP_SIZE = 32;

// SYCL: warp_reduce using sub-group shuffle operations
template <typename acc_t, int WARP_BATCH, int WARP_SIZE_T, template<typename> class ReduceOp>
inline void warp_reduce(acc_t* sum, sycl::sub_group sg) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE_T / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < WARP_BATCH; ++i) {
            acc_t b = sycl::shift_group_left(sg, sum[i], offset);
            sum[i] = r(sum[i], b);
        }
    }
}

// SYCL softmax kernels would be submitted via sycl::queue::submit
// The template parameters and logic mirror the CUDA version but use
// sycl::nd_item for work-item indexing and sycl::sub_group for shuffles.

} // anonymous namespace
