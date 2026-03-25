// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/Sorting.cpp

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/opencl/Sorting.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/Context.h>
#include <ATen/TensorUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/sycl/detail/TensorInfo.h>

#include <ATen/native/SortingUtils.h>
#include <ATen/native/ReduceOpsUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/full.h>
#include <ATen/ops/kthvalue_native.h>
#include <ATen/ops/median_native.h>
#include <ATen/ops/nanmedian_native.h>
#include <ATen/ops/where.h>
#include <ATen/ops/rsub.h>
#include <ATen/ops/div.h>
#include <ATen/ops/index.h>
#endif

namespace at::native {
namespace {

std::tuple<Tensor&, Tensor&> kthvalue_out_impl_opencl(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  int64_t slicesize = self.dim() == 0 ? 1 : self.size(dim);
  zero_numel_check_dims(self, dim, "kthvalue()");

  TORCH_CHECK(k >= 1 && k <= slicesize,
              "kthvalue(): selected number k out of range for dimension ", dim);

  TORCH_CHECK(
      slicesize <= std::numeric_limits<int32_t>::max(),
      "kthvalue(): dimension ", dim, " is too large (", slicesize,
      "). The current OpenCL implementation supports dimension sizes up to ",
      std::numeric_limits<int32_t>::max());

  at::assert_no_overlap(self, values);

  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim, keepdim);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }

  TORCH_CHECK(
      self.dim() <= MAX_TENSORINFO_DIMS,
      "cannot operate on more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  if (self.numel() != 0) {
    launch_kthvalue_kernel(values, indices, self, dim, k);
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor&, Tensor&> median_with_indices_impl(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    bool ignore_nan) {
  // See note [Writing Nondeterministic Operations]
  at::globalContext().alertNotDeterministic("median OpenCL with indices output");
  NoNamesGuard guard;

  dim = at::maybe_wrap_dim(dim, self.dim());
  Tensor in = self.dim() > 0 ? self.contiguous() : self.unsqueeze(0);

  checkDeviceType("median", {values, indices}, self.device().type());
  checkScalarType("median", {indices, "indices", 1}, kLong);
  checkSameType("median", {values, "values", 0}, {self, "self", 2});

  TORCH_CHECK(
      self.dim() <= MAX_TENSORINFO_DIMS,
      "median() cannot operate on more than ",
      MAX_TENSORINFO_DIMS,
      " dimensions");

  std::vector<int64_t> out_shape = self.sizes().vec();
  zero_numel_check_dims(self, dim, "median()");
  if (self.dim() > 0) {
    assert(dim >= 0);
    assert(dim < static_cast<int64_t>(out_shape.size()));

    if (keepdim) {
      out_shape[dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + dim);
    }
  }

  values.resize_(out_shape);
  indices.resize_(out_shape);

  if (self.numel() > 0) {
    Tensor vals = keepdim && self.dim() > 0 ? values : values.unsqueeze(dim);
    Tensor inds = keepdim && self.dim() > 0 ? indices : indices.unsqueeze(dim);

    launch_median_kernel(vals, inds, in, dim, ignore_nan);
  }

  guard.reset();
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);

  return std::forward_as_tuple(values, indices);
}

Tensor median_impl(const Tensor& self, bool ignore_nan) {
  NoNamesGuard guard;

  int64_t size = self.numel();
  if (size <= 0) {
    return at::full({}, std::numeric_limits<float>::quiet_NaN()).to(self.options());
  }

  Tensor sorted = std::get<0>(self.flatten().sort());

  if (!ignore_nan) {
    int64_t k = (size - 1) / 2;
    return at::where(sorted[-1].isnan(), sorted[-1], sorted[k]);
  } else {
    Tensor k = at::div(at::rsub(sorted.isnan().sum(), (size - 1)), 2).to(kLong);
    return at::index(sorted, {k});
  }
}

} // namespace (anonymous)

std::tuple<Tensor&, Tensor&> kthvalue_out_opencl(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  auto result = [&]() {
    NoNamesGuard guard;
    return kthvalue_out_impl_opencl(values, indices, self.contiguous(), k, dim, keepdim);
  }();
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
  return result;
}

// Mark: median

std::tuple<Tensor&, Tensor&> median_out_opencl(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return median_with_indices_impl(
      values, indices, self, dim, keepdim, /*ignore_nan=*/false);
}

Tensor median_opencl(const Tensor& self) {
  return median_impl(self, /*ignore_nan=*/false);
}

std::tuple<Tensor&, Tensor&> nanmedian_out_opencl(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return median_with_indices_impl(
      values, indices, self, dim, keepdim, /*ignore_nan=*/true);
}

Tensor nanmedian_opencl(const Tensor& self) {
  return median_impl(self, /*ignore_nan=*/true);
}

} // namespace at::native
