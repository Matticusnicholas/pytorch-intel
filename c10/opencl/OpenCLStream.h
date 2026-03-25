// AI-TRANSLATED: OpenCL/SYCL stream management for Intel GPU support
#pragma once

#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/opencl/OpenCLMacros.h>

// SYCL TODO: needs_review - In production, this should wrap sycl::queue objects
// #include <sycl/sycl.hpp>

namespace c10::opencl {

class C10_OPENCL_API OpenCLStream {
 public:
  enum Unchecked { UNCHECKED };

  explicit OpenCLStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::OPENCL);
  }

  OpenCLStream(Unchecked, Stream stream) : stream_(stream) {}

  bool operator==(const OpenCLStream& other) const noexcept {
    return stream_ == other.stream_;
  }

  bool operator!=(const OpenCLStream& other) const noexcept {
    return stream_ != other.stream_;
  }

  operator Stream() const { return stream_; }

  DeviceType device_type() const { return DeviceType::OPENCL; }
  DeviceIndex device_index() const { return stream_.device_index(); }
  Device device() const { return Device(DeviceType::OPENCL, device_index()); }
  StreamId id() const { return stream_.id(); }

  // SYCL TODO: needs_review - return actual sycl::queue&
  // sycl::queue& queue() const;

  void synchronize() const {
    // SYCL TODO: needs_review - call queue().wait_and_throw()
  }

 private:
  Stream stream_;
};

C10_OPENCL_API OpenCLStream getStreamFromPool(
    bool isHighPriority = false,
    DeviceIndex device_index = -1);

C10_OPENCL_API OpenCLStream getDefaultOpenCLStream(
    DeviceIndex device_index = -1);

C10_OPENCL_API OpenCLStream getCurrentOpenCLStream(
    DeviceIndex device_index = -1);

C10_OPENCL_API void setCurrentOpenCLStream(OpenCLStream stream);

} // namespace c10::opencl
