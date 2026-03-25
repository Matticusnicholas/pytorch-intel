// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/TensorShapeCUDA.cpp

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/native/Resize.h>
#include <ATen/native/opencl/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/set_native.h>
#endif

namespace at::native {

Tensor& set_opencl_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      at::opencl::getOpenCLDeviceAllocator(),
      true);
  result.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

Tensor& set_storage_opencl_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  checkSetStorage(result, std::move(storage), storage_offset, size, stride);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ?
                                          at::OptionalIntArrayRef(stride) : std::nullopt;
  at::native::resize_impl_opencl_(result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}

} // namespace at::native
