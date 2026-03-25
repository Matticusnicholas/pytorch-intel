// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/Equal.cpp

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#include <ATen/OpenCLFunctions.h>
#else
#include <ATen/ops/eq_opencl_dispatch.h>
#include <ATen/ops/equal_native.h>
#endif

namespace at::native {

bool opencl_equal(const Tensor& self, const Tensor &src) {
  if (!at::namedinference::are_names_equal(
          self.unsafeGetTensorImpl(), src.unsafeGetTensorImpl())) {
    return false;
  }
  at::NoNamesGuard guard;
  TORCH_CHECK(self.device() == src.device(), "Cannot compare two tensors on "
              "different devices. Got: ", self.device(), " and ", src.device());
  if (self.sizes() != src.sizes()) {
    return false;
  }
  if (self.numel() == 0) {
    return true;
  }

  if (self.is_alias_of(src)
      && self.storage_offset() == src.storage_offset()
      && self.dtype() == src.dtype()
      && self.is_contiguous() == src.is_contiguous()
      && self.strides().equals(src.strides())
      && self.layout() == src.layout()
      && self.is_neg() == src.is_neg()
      && self.is_conj() == src.is_conj()) {
    return true;
  }

  return at::opencl::eq(self, src).all().item().to<bool>();
}

} // namespace at::native
