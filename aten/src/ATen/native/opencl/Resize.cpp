// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/Resize.cpp

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/opencl/Resize.h>
#include <ATen/core/Tensor.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/native/ResizeCommon.h>
#include <c10/opencl/OpenCLGuard.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/resize_native.h>
#endif

namespace at::native {

void resize_bytes_opencl(StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");
  auto allocator = storage->allocator();
  TORCH_CHECK(allocator != nullptr, "Trying to resize storage without an allocator");

  c10::Device device = storage->device();

  if (size_bytes == 0) {
    storage->set_data_ptr_noswap(at::DataPtr(nullptr, device));
    storage->set_nbytes(0);
    return;
  }

  c10::opencl::OpenCLGuard guard(device.index());
  at::DataPtr data = allocator->allocate(size_bytes);
  if (storage->data_ptr()) {
    at::globalContext().lazyInitDevice(c10::DeviceType::OpenCL);

    // SYCL TODO: needs_review - replace with SYCL memcpy equivalent
    // In CUDA this uses cudaMemcpyAsync with cudaMemcpyDeviceToDevice.
    // For SYCL, use queue.memcpy() from the current OpenCL stream/queue.
    at::opencl::memcpyAsync(
        data.get(),
        storage->data(),
        std::min(storage->nbytes(), size_bytes),
        at::opencl::getCurrentOpenCLStream());
  }

  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(size_bytes);
}

const Tensor& resize_opencl_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* self_ = self.unsafeGetTensorImpl();
  auto old_storage_nbytes = self_->unsafe_storage() ? self_->unsafe_storage().nbytes() : 0;
  resize_impl_opencl_(self_, size, /*stride=*/std::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_resize_deterministic_(self, static_cast<int64_t>(old_storage_nbytes));
  }
  return self;
}
} // namespace at::native
