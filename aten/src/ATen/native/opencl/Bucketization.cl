// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/Bucketization.cu
// SYCL TODO: needs_review - complex kernel requires manual optimization for Intel GPU
// Contains raw CUDA kernel with <<<>>> launch, __device__ binary search functions.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/native/BucketizationUtils.h>
#include <ATen/native/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/bucketize_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/searchsorted_native.h>
#endif

namespace at::native {

namespace {

template<typename input_t>
int64_t lower_bound_sycl(const input_t *data_ss, int64_t start, int64_t end, const input_t val, const int64_t *data_sort) {
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_sort ? data_ss[orig_start + data_sort[mid]] : data_ss[mid];
    if (!(mid_val >= val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t>
int64_t upper_bound_sycl(const input_t *data_ss, int64_t start, int64_t end, const input_t val, const int64_t *data_sort) {
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_sort ? data_ss[orig_start + data_sort[mid]] : data_ss[mid];
    if (!(mid_val > val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template<typename input_t, typename output_t>
void searchsorted_sycl_contiguous(Tensor& result, const Tensor& input, const Tensor& boundaries, const bool& right, const Tensor& sorter) {
  int64_t numel_in = input.numel();
  bool is_scalar_input = input.dim() == 0 && numel_in == 1;
  int64_t idim_in = is_scalar_input ? 1 : input.sizes().back();
  int64_t idim_bd = boundaries.sizes().back();

  const input_t *data_in = input.const_data_ptr<input_t>();
  const input_t *data_bd = boundaries.const_data_ptr<input_t>();
  const int64_t *data_sort = sorter.defined() ? sorter.const_data_ptr<int64_t>() : nullptr;
  output_t *data_out = result.mutable_data_ptr<output_t>();

  auto& queue = at::sycl::getCurrentSYCLQueue();
  int64_t maxThread = 256;
  int64_t maxGrid = 1024;
  int64_t block_x = std::min(maxThread, numel_in);
  int64_t grid_x = std::min(maxGrid, ceil_div<int64_t>(numel_in, block_x));
  bool is_1d = boundaries.dim() == 1;

  queue.parallel_for(
      sycl::nd_range<1>(grid_x * block_x, block_x),
      [=](sycl::nd_item<1> item) {
        for (int64_t tid = item.get_global_id(0); tid < numel_in;
             tid += item.get_global_range(0)) {
          int64_t start_bd = is_1d ? 0 : tid / idim_in * idim_bd;
          int64_t end_bd = start_bd + idim_bd;

          int64_t pos = !right ?
            lower_bound_sycl<input_t>(data_bd, start_bd, end_bd, data_in[tid], data_sort) - start_bd :
            upper_bound_sycl<input_t>(data_bd, start_bd, end_bd, data_in[tid], data_sort) - start_bd;

          data_out[tid] = pos;
        }
      });
}

void dispatch(Tensor& result, const Tensor& input, const Tensor& boundaries,
              bool out_int32, bool right, const Tensor& sorter) {
  if (!out_int32) {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "searchsorted_out_opencl", [&] {
      searchsorted_sycl_contiguous<scalar_t, int64_t>(result, input, boundaries, right, sorter);
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "searchsorted_out_opencl", [&] {
      searchsorted_sycl_contiguous<scalar_t, int>(result, input, boundaries, right, sorter);
    });
  }
}

} // namespace

Tensor& searchsorted_out_opencl(const Tensor& sorted_sequence, const Tensor& self,
    bool out_int32, bool right, const std::optional<std::string_view> side_opt,
    const std::optional<Tensor>& sorter_opt, Tensor& result) {
  c10::MaybeOwned<Tensor> sorter_maybe_owned = at::borrow_from_optional_tensor(sorter_opt);
  const Tensor& sorter = *sorter_maybe_owned;
  searchsorted_pre_check(sorted_sequence, self, result, out_int32, right, side_opt, sorter);
  resize_output(result, self.sizes());

  bool is_right = (side_opt && *side_opt == "right") || right;
  if (self.numel() == 0) return result;

  Tensor out = result;
  if (!result.is_contiguous()) out = result.contiguous();

  if (sorted_sequence.is_contiguous() && self.is_contiguous() && sorted_sequence.dtype() == self.dtype() && sorter.is_contiguous()) {
    dispatch(out, self, sorted_sequence, out_int32, is_right, sorter);
  } else {
    Tensor trimmed_input, trimmed_boundaries, trimmed_sorter;
    searchsorted_maybe_trim_input_tensors(trimmed_input, trimmed_boundaries, trimmed_sorter, self, sorted_sequence, sorter);
    const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
    const Tensor& final_boundaries = trimmed_boundaries.defined() ? trimmed_boundaries : sorted_sequence;
    const Tensor& final_sorter = trimmed_sorter.defined() ? trimmed_sorter : sorter;
    dispatch(out, final_input, final_boundaries, out_int32, is_right, final_sorter);
  }

  if (!result.is_contiguous()) result.copy_(out);
  return result;
}

Tensor& searchsorted_out_opencl(const Tensor& sorted_sequence, const Scalar& self,
    bool out_int32, bool right, const std::optional<std::string_view> side_opt,
    const std::optional<Tensor>& sorter_opt, Tensor& result) {
  const Tensor& scalar_tensor = searchsorted_scalar_tensor(self, sorted_sequence.device());
  return searchsorted_out_opencl(sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter_opt, result);
}

Tensor searchsorted_opencl(const Tensor& sorted_sequence, const Tensor& self,
    bool out_int32, bool right, const std::optional<std::string_view> side_opt,
    const std::optional<Tensor>& sorter) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::native::searchsorted_out_opencl(sorted_sequence, self, out_int32, right, side_opt, sorter, result);
  return result;
}

Tensor searchsorted_opencl(const Tensor& sorted_sequence, const Scalar& self,
    bool out_int32, bool right, const std::optional<std::string_view> side_opt,
    const std::optional<Tensor>& sorter) {
  const Tensor& scalar_tensor = searchsorted_scalar_tensor(self, sorted_sequence.device());
  return searchsorted_opencl(sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter);
}

Tensor& bucketize_out_opencl(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right, Tensor& result) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  at::native::searchsorted_out_opencl(boundaries, self, out_int32, right, std::nullopt, std::nullopt, result);
  return result;
}

Tensor bucketize_opencl(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options = TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  at::native::bucketize_out_opencl(self, boundaries, out_int32, right, result);
  return result;
}

Tensor bucketize_opencl(const Scalar& self, const Tensor& boundaries, bool out_int32, bool right) {
  return bucketize_opencl(searchsorted_scalar_tensor(self, boundaries.device()), boundaries, out_int32, right);
}

} // namespace at::native
