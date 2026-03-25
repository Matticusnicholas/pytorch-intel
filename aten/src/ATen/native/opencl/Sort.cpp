// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/Sort.cpp

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/opencl/Sort.h>
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Sorting.h>
#include <ATen/native/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/sort_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <limits>

namespace at::native {

std::vector<int64_t> infer_dense_strides_dim_last(const Tensor & self, int64_t dim);

void fillSliceWithIndex(const Tensor& t, int64_t dim) {
  if (t.numel()) {
    auto sizes = DimVector(t.dim(), 1);
    sizes[dim] = t.sizes()[dim];
    auto range = at::arange(t.sizes()[dim], t.options());
    auto rangeview = range.view(sizes);
    t.copy_(rangeview);
  }
}

void sort_opencl_kernel(
    const TensorBase& self_base,
    const TensorBase& values_base,
    const TensorBase& indices_base,
    int64_t dim,
    bool descending,
    bool stable) {
  // this algorithm is always stable

#define TOTENSOR(BASE, VAR)           \
  OptionalTensorRef opt_##BASE(BASE); \
  const Tensor& VAR = *opt_##BASE;

  TOTENSOR(self_base, self);
  TOTENSOR(values_base, values);
  TOTENSOR(indices_base, indices);

  TORCH_CHECK(self.sizes()[dim] <= std::numeric_limits<int>::max(),
    "The dimension being sorted can not have more than INT_MAX elements.");

  const auto self_dtype = self.dtype();
  TORCH_CHECK(self_dtype != ScalarType::ComplexFloat && self_dtype != ScalarType::ComplexDouble,
    "Sort currently does not support complex dtypes on OpenCL.");

  // use inplace algorithm for smaller input sizes without stable=True
  if (should_use_small_sort(self, dim)) {
    fillSliceWithIndex(indices, dim);
    values.copy_(self);
    sortKeyValueInplace(values, indices, dim, descending, stable);
    return;
  }

  Tensor self_;
  bool newself = false;
  if (self.is_non_overlapping_and_dense() && self.stride(dim) == 1) {
    self_ = self;
  } else {
    auto new_strides_unsort = infer_dense_strides_dim_last(self, dim);
    self_ = at::empty_strided(self.sizes(), new_strides_unsort, self.options());
    self_.copy_(self);
    newself = true;
  }

  c10::MaybeOwned<Tensor> values_tmp, indices_tmp;
  if (values.strides() == self_.strides() && (newself || get_overlap_status(self, values) == MemOverlapStatus::No)) {
    values_tmp = c10::MaybeOwned<Tensor>::borrowed(values);
  } else {
    values_tmp = c10::MaybeOwned<Tensor>::owned(
        at::empty_strided(self_.sizes(), self_.strides(), self_.options()));
  }

  if (indices.strides() != self_.strides()) {
    indices_tmp = c10::MaybeOwned<Tensor>::owned(
        at::empty_strided(self_.sizes(), self_.strides(), self_.options().dtype(kLong)));
  } else {
    indices_tmp = c10::MaybeOwned<Tensor>::borrowed(indices);
  }

  launch_stable_sort_kernel(self_, dim, descending, *values_tmp, *indices_tmp);

  if (!values_tmp->is_same(values)) {
    values.copy_(*values_tmp);
  }
  if (!indices_tmp->is_same(indices)) {
    indices.copy_(*indices_tmp);
  }
}

REGISTER_OPENCL_DISPATCH(sort_stub, &sort_opencl_kernel)

}  // namespace at::native
