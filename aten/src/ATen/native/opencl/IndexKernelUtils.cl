// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/IndexKernelUtils.cu

#include <ATen/sycl/SYCLContext.h>
#include <ATen/native/sycl/MemoryAccess.h>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/ceil_div.h>

namespace at::native {
template <int Alignment, typename index_t>
/* SYCL kernel */ void vectorized_gather_kernel(char * out, char * inp, index_t * idx, int num_ind, int64_t slice_size, int64_t ind_dim_size, int64_t inp_stride, int64_t out_stride, bool allow_neg_indices) {
    int64_t ind = idx[item.get_group(0)];
    if (allow_neg_indices) {
        ind = (ind < 0) ? ind + ind_dim_size : ind;
    }
    SYCL_KERNEL_ASSERT_VERBOSE(ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds", "Expected 0 <= index < ind_dim_size(%ld), but got index = %ld", ind_dim_size, ind);
    // off is guaranteed to be within int32 limits
    for (int32_t off = (item.get_local_range(0) * item.get_group(1) + item.get_local_id(0)) * Alignment; off < slice_size; off += item.get_local_range(0) * item.get_group_range(1) * Alignment) {
      auto vec = at::native::memory::ld_vec<Alignment>(inp + ind * inp_stride + off);
      at::native::memory::st_vec<Alignment>(out + item.get_group(0) * (int32_t)out_stride + off, vec);  // out offset is guaranteed to be within int32 limits
    }
}



template <int64_t Alignment, typename index_t>
void vectorized_gather_kernel_launch(char * out, char * inp, index_t * idx, int num_ind,
                                     int64_t slice_size_in_bytes, int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes, bool allow_neg_indices){

  constexpr int64_t max_num_threads=256;
  auto num_threads = at::round_up(
      at::ceil_div(slice_size_in_bytes, Alignment),
      static_cast<int64_t>(SYCL_SUB_GROUP_SIZE));
  uint32_t grid_y = at::sycl::getCurrentDeviceProperties()->maxGridSize[1];
  grid_y = std::min(static_cast<uint32_t>(at::ceil_div(slice_size_in_bytes, max_num_threads * Alignment)), grid_y);
  sycl::range<3> grid = {static_cast<uint32_t>(num_ind), grid_y, 1};
  auto block = std::min(max_num_threads, num_threads);
  vectorized_gather_kernel<Alignment, index_t>/* SYCL: kernel launch with nd_range(grid, block, 0, at::sycl::getCurrentSYCLStream()) */(out, inp, idx, num_ind, slice_size_in_bytes,
  ind_dim_size, inp_stride_bytes, out_stride_bytes, allow_neg_indices);
  // SYCL: kernel launch check handled by SYCL runtime;
}

// explicit template instantiation
template void vectorized_gather_kernel_launch<16, int64_t>(char * out, char * inp, int64_t * idx, int num_ind, int64_t slice_size_in_bytes,
int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes, bool allow_neg_indices);
template void vectorized_gather_kernel_launch<16, int32_t>(char * out, char * inp, int32_t * idx, int num_ind, int64_t slice_size_in_bytes,
int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes, bool allow_neg_indices);

}
