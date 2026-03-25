#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/sycl/Sorting.h>
#include <ATen/core/TensorBase.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <c10/macros/Macros.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/sycl/detail/TensorInfo.h>
#include <ATen/native/sycl/SortingCommon.h>
#include <ATen/native/sycl/SortingRadixSelect.h>

#include <c10/sycl/SYCLStream.h>

#include <cassert>
#include <cstdlib>

namespace at::native {

namespace {

// Finds the rank k element, and its index, of the values along dimension dim
template <typename scalar_t, typename index_t, int Dim>
// SYCL kernel
void gatherKthValue(
    sycl::detail::TensorInfo<const scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t k,
    index_t numInputSlices,
    index_t inputWithinSliceStride,
    sycl::detail::TensorInfo<scalar_t, index_t> kthValue,
    sycl::detail::TensorInfo<int64_t, index_t> indices) {
  // smem is used by radixSelect for radix bin counts. Type must be index_t to
  // handle sliceSize > INT_MAX.
#if 1 // SYCL: not ROCm
  // SYCL: use sycl::local_accessor index_t smem[C10_WARP_SIZE]; // one per each warp, up to warp limit
#else
  // Maximum shared memory size for radix select (used in countRadixAggregateCounts): NUM_BUFFERS * MAX_WARPS * RADIX_SIZE.
  // HIP workgroups have at most 1024 threads. Warp size is at least 32 (can be 64 on some
  // architectures), so we use 32 for safety: 2 buffers * (1024/32) warps * 4 radix bins = 256.
  // SYCL: use sycl::local_accessor index_t smem[256];
#endif

  index_t slice = getLinearBlockId<index_t>();
  if (slice >= numInputSlices) {
    return;
  }

  // Find the start offset for our slice
  index_t sliceStartIndex =
      sycl::detail::IndexToOffset<const scalar_t, index_t, Dim>::get(slice, input);
  index_t kthValueSliceStartIndex =
      sycl::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, kthValue);
  index_t indicesSliceStartIndex =
      sycl::detail::IndexToOffset<int64_t, index_t, Dim>::get(slice, indices);

  const scalar_t* inputSliceStart = &input.data[sliceStartIndex];
  scalar_t* kthValueSliceStart = &kthValue.data[kthValueSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];

  // Find the k-th highest element in our input
  scalar_t kValue = static_cast<scalar_t>(0);
  radixSelect<
      scalar_t,
      typename TopKTypeConfig<scalar_t>::RadixType,
      index_t>(
      inputSliceStart,
      k,
      false,
      inputSliceSize,
      inputWithinSliceStride,
      smem,
      &kValue);

  // Find the index of the k-th highest element
  // SYCL: use sycl::local_accessor int32_t minIndexFound;

  if (item.get_local_id(0) == 0) {
      minIndexFound = static_cast<int32_t>(inputSliceSize);
  }
  item.barrier(sycl::access::fence_space::local_space);

  for (index_t i = item.get_local_id(0); i < inputSliceSize; i += item.get_local_range(0)) {
      // Early exit based on best-so-far
      if (i >= minIndexFound) {
          break;
      }

      scalar_t v = doLdg(&inputSliceStart[i * inputWithinSliceStride]);
      bool isKValue =
          ((v == kValue) || (at::_isnan(v) && at::_isnan(kValue)));

      if (isKValue) {
          sycl::atomic_ref_min /* sycl::atomic_ref */(&minIndexFound, static_cast<int32_t>(i));
          break;
      }
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (item.get_local_id(0) == 0) {
      indicesSliceStart[0] = static_cast<index_t>(minIndexFound);
      kthValueSliceStart[0] = kValue;
  }
}

// CUDA kernel to find the median, and its index, of the values along dimension dim
template <typename scalar_t, typename index_t, int Dim>
// SYCL kernel
void gatherMedian(
    sycl::detail::TensorInfo<scalar_t, index_t> values,
    sycl::detail::TensorInfo<int64_t, index_t> indices,
    sycl::detail::TensorInfo<const scalar_t, index_t> input,
    index_t inputSliceSize,
    index_t numInputSlices,
    index_t inputWithinSliceStride,
    bool ignore_nan) {
  // smem is used by radixSelect for radix bin counts. Type must be index_t to
  // handle sliceSize > INT_MAX.
#if 1 // SYCL: not ROCm
  // SYCL: use sycl::local_accessor index_t smem[C10_WARP_SIZE]; // one per each warp, up to warp limit
#else
  // Maximum shared memory size for radix select (used in countRadixAggregateCounts): NUM_BUFFERS * MAX_WARPS * RADIX_SIZE.
  // HIP workgroups have at most 1024 threads. Warp size is at least 32 (can be 64 on some
  // architectures), so we use 32 for safety: 2 buffers * (1024/32) warps * 4 radix bins = 256.
  // SYCL: use sycl::local_accessor index_t smem[256];
#endif

  index_t slice = getLinearBlockId<index_t>();
  if (slice >= numInputSlices) {
    return;
  }

  // Finds the start offset for our slice
  index_t valuesSliceStartIndex =
      sycl::detail::IndexToOffset<scalar_t, index_t, Dim>::get(slice, values);
  index_t indicesSliceStartIndex =
      sycl::detail::IndexToOffset<int64_t, index_t, Dim>::get(slice, indices);
  index_t inputSliceStartIndex =
      sycl::detail::IndexToOffset<const scalar_t, index_t, Dim>::get(slice, input);

  scalar_t* valuesSliceStart = &values.data[valuesSliceStartIndex];
  int64_t* indicesSliceStart = &indices.data[indicesSliceStartIndex];
  const scalar_t* inputSliceStart = &input.data[inputSliceStartIndex];

  index_t nan_count = 0;
  for (index_t i = item.get_local_id(0); i < inputSliceSize; i += item.get_local_range(0)) {
    scalar_t val = doLdg(&inputSliceStart[i * inputWithinSliceStride]);
    nan_count += at::_isnan(val) ? 1 : 0;
  }

  // Counts number of nan values
  // This code performs a parallel sum reduction (not the most efficient code)
  // SYCL: use sycl::local_accessor int64_t num_nan;
  if (item.get_local_id(0) == 0) {
    num_nan = 0;
  }
  item.barrier(sycl::access::fence_space::local_space);
  if (nan_count > 0) {
    gpuAtomicAddNoReturn(&num_nan, nan_count);
  }
  item.barrier(sycl::access::fence_space::local_space);

  // For torch.median, if we found nan set k to last index so the computed value
  // is nan, otherwise set k to the middle element of the non-nan values
  index_t k = (!ignore_nan && num_nan > 0) ? inputSliceSize - 1
                                           : (inputSliceSize - num_nan - 1) / 2;

  // Find the median
  scalar_t median = static_cast<scalar_t>(0);
  radixSelect<
      scalar_t,
      typename TopKTypeConfig<scalar_t>::RadixType,
      index_t>(
      inputSliceStart,
      k + 1,
      false,
      inputSliceSize,
      inputWithinSliceStride,
      smem,
      &median);

  valuesSliceStart[0] = median;

  // Find the index of the median value in the slice
  for (index_t i = item.get_local_id(0); i < inputSliceSize; i += item.get_local_range(0)) {
    scalar_t val = doLdg(&inputSliceStart[i * inputWithinSliceStride]);
    if (val == median || (at::_isnan(val) && at::_isnan(median))) {
      indicesSliceStart[0] = i;
      break;
    }
  }
}

struct KthValueLauncher {
  int64_t k;

  KthValueLauncher(int64_t k) : k(k) {}

  template <typename scalar_t, typename index_t, int all_dims>
  inline void launch(
      sycl::detail::TensorInfo<scalar_t, index_t> values_info,
      int collapse_values_dim,
      sycl::detail::TensorInfo<int64_t, index_t> indices_info,
      [[maybe_unused]] int collapse_indices_dim,
      sycl::detail::TensorInfo<const scalar_t, index_t> self_info,
      int collapse_self_dim,
      int64_t num_slices,
      int64_t slice_size) {
    sycl::range<3> grid;
    if (!getGridFromTiles(num_slices, grid)) {
      TORCH_CHECK(false, "slices are too many");
    }

    sycl::range<3> block(std::min(
        round_up(slice_size, (int64_t)at::sycl::warp_size()), (int64_t)1024));
    auto stream = at::sycl::getCurrentSYCLStream();
    gatherKthValue<scalar_t, index_t, all_dims>/* SYCL: launch with nd_range(grid, block, 0, stream) */(
        self_info,
        slice_size,
        k,
        num_slices,
        /* The actual dimension that the k-selection is running in */
        /* may have changed from collapseDims() */
        self_info.strides[collapse_self_dim],
        values_info,
        indices_info);
    // SYCL: kernel launch check handled by SYCL runtime;
  }
};

struct MedianLauncher {
  bool ignore_nan;

  MedianLauncher(bool ignore_nan) : ignore_nan(ignore_nan) {}

  template <typename scalar_t, typename index_t, int all_dims>
  inline void launch(
      sycl::detail::TensorInfo<scalar_t, index_t> values_info,
      [[maybe_unused]] int collapse_values_dim,
      sycl::detail::TensorInfo<int64_t, index_t> indices_info,
      [[maybe_unused]] int collapse_indices_dim,
      sycl::detail::TensorInfo<const scalar_t, index_t> self_info,
      int collapse_self_dim,
      int64_t num_slices,
      int64_t slice_size) {
    sycl::range<3> grid;
    if (!getGridFromTiles(num_slices, grid)) {
      TORCH_CHECK(false, "slices are too many");
    }

    sycl::range<3> block(std::min(
        round_up(slice_size, (int64_t)at::sycl::warp_size()), (int64_t)1024));
    auto stream = at::sycl::getCurrentSYCLStream();
    gatherMedian<scalar_t, index_t, all_dims>/* SYCL: launch with nd_range(grid, block, 0, stream) */(
        values_info,
        indices_info,
        self_info,
        slice_size,
        num_slices,
        self_info.strides[collapse_self_dim],
        ignore_nan);
    // SYCL: kernel launch check handled by SYCL runtime;
  }
};

}  // namespace (anonymous)

void launch_kthvalue_kernel(
    const TensorBase &values, const TensorBase &indices,
    const TensorBase &self, int64_t dim, int64_t k) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "kthvalue_opencl", [&] {
    AT_DISPATCH_INDEX_TYPES(
        sycl::detail::canUse32BitIndexMath(self) &&
        sycl::detail::canUse32BitIndexMath(values) &&
        sycl::detail::canUse32BitIndexMath(indices) ? ScalarType::Int : ScalarType::Long,
        "kth_value_launcher", [&] {
          run_launcher<scalar_t, index_t>(
              values, indices, self, dim, KthValueLauncher(k));
    });
  });
}

void launch_median_kernel(
    const TensorBase &vals, const TensorBase &inds,
    const TensorBase &self, int64_t dim, bool ignore_nan) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "median_out_impl", [&] {
        if (sycl::detail::canUse32BitIndexMath(vals) &&
            sycl::detail::canUse32BitIndexMath(inds) &&
            sycl::detail::canUse32BitIndexMath(self)) {
          run_launcher<scalar_t, uint32_t>(
              vals, inds, self, dim, MedianLauncher(ignore_nan));
        } else {
          run_launcher<scalar_t, uint64_t>(
              vals, inds, self, dim, MedianLauncher(ignore_nan));
        }
      });
}

} // namespace at::native
