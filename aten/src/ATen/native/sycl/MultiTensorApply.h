// AI-TRANSLATED: SYCL equivalent of CUDA header for Intel GPU support
// Original: ATen/native/cuda/MultiTensorApply.cuh
#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/sycl/MemoryAccess.h>
#include <vector>

namespace at::native {

namespace {

static constexpr int64_t kILP = 4;
static constexpr int64_t kChunkSize = 65536;
static constexpr int64_t kBlockSize = 512;

static constexpr int depth_to_max_tensors[5] = {110, 64, 48, 36, 30};
static constexpr int depth_to_max_blocks[5] = {320, 320, 320, 320, 320};
static constexpr int depth_to_max_tensors_scalarlist[5] = {96, 64, 48, 36, 30};
static constexpr int depth_to_max_tensors_scalarlist_of_complex_double[2] = {72, 60};

template <typename T>
inline bool is_aligned(T* p) {
  return ((uint64_t)p) % (kILP * sizeof(T)) == 0;
}

template <typename T>
inline void load_store(
    T* dst,
    T* src,
    int64_t dst_offset,
    int64_t src_offset) {
  using LT = at::native::memory::aligned_vector<T, kILP>;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

template <int n>
struct TensorListMetadata {
  const void* addresses[n][depth_to_max_tensors[n - 1]];
  int64_t numel_for_tensor[depth_to_max_tensors[n - 1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
  int block_to_chunk[depth_to_max_blocks[n - 1]];
  int start_tensor_this_launch;
};

template <typename scalar_vals_t, int n>
struct TensorListScalarListMetadata {
  const void* addresses[n][depth_to_max_tensors_scalarlist[n - 1]];
  int64_t numel_for_tensor[depth_to_max_tensors_scalarlist[n - 1]];
  scalar_vals_t scalar_vals[depth_to_max_tensors_scalarlist[n - 1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
  int block_to_chunk[depth_to_max_blocks[n - 1]];
};

template <>
struct TensorListScalarListMetadata<c10::complex<double>, 1> {
  const void* addresses[1][depth_to_max_tensors_scalarlist_of_complex_double[0]];
  int64_t numel_for_tensor[depth_to_max_tensors_scalarlist_of_complex_double[0]];
  c10::complex<double> scalar_vals[depth_to_max_tensors_scalarlist_of_complex_double[0]];
  unsigned char block_to_tensor[depth_to_max_blocks[0]];
  int block_to_chunk[depth_to_max_blocks[0]];
};

template <>
struct TensorListScalarListMetadata<c10::complex<double>, 2> {
  const void* addresses[2][depth_to_max_tensors_scalarlist_of_complex_double[1]];
  int64_t numel_for_tensor[depth_to_max_tensors_scalarlist_of_complex_double[1]];
  c10::complex<double> scalar_vals[depth_to_max_tensors_scalarlist_of_complex_double[1]];
  unsigned char block_to_tensor[depth_to_max_blocks[1]];
  int block_to_chunk[depth_to_max_blocks[1]];
};

template <int n>
struct FusedOptimizerTensorListMetadata {
  const void* addresses[n][depth_to_max_tensors[n - 1]];
  int64_t numel_for_tensor[depth_to_max_tensors[n - 1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
  int block_to_chunk[depth_to_max_blocks[n - 1]];
  const void* state_steps_addresses[depth_to_max_tensors[n - 1]];
};

// SYCL: multi_tensor_apply launches SYCL kernels via sycl::queue::submit
// instead of CUDA kernel<<<grid, block>>>
template <int depth, typename T, typename... ArgTypes>
void multi_tensor_apply(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    at::TensorList state_steps,
    T callable,
    ArgTypes... args);

} // namespace

} // namespace at::native
