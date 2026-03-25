// AI-TRANSLATED: OpenCL guard implementation for Intel GPU support
#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/opencl/OpenCLMacros.h>

namespace c10::opencl::impl {

struct OpenCLGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::OPENCL;

  OpenCLGuardImpl() = default;
  explicit OpenCLGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::OPENCL);
  }

  DeviceType type() const override {
    return DeviceType::OPENCL;
  }

  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == DeviceType::OPENCL);
    auto old_index = current_device_;
    current_device_ = d.index();
    return Device(DeviceType::OPENCL, old_index);
  }

  Device getDevice() const override {
    return Device(DeviceType::OPENCL, current_device_);
  }

  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == DeviceType::OPENCL);
    current_device_ = d.index();
  }

  void uncheckedSetDevice(Device d) const noexcept override {
    current_device_ = d.index();
  }

  Stream getStream(Device d) const noexcept override {
    return Stream(Stream::DEFAULT, d);
  }

  Stream getDefaultStream(Device d) const override {
    return Stream(Stream::DEFAULT, d);
  }

  Stream exchangeStream(Stream) const noexcept override {
    // OpenCL stream management - minimal implementation
    return Stream(Stream::DEFAULT, Device(DeviceType::OPENCL, current_device_));
  }

  DeviceIndex deviceCount() const noexcept override {
    // SYCL TODO: needs_review - implement actual device enumeration via SYCL runtime
    return 1;
  }

private:
  // SYCL TODO: needs_review - this should be thread_local in production
  mutable DeviceIndex current_device_ = 0;
};

} // namespace c10::opencl::impl
