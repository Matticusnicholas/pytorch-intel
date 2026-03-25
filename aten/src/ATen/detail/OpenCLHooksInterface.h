// AI-TRANSLATED: OpenCL hooks interface for Intel GPU (Arc/Xe) support
#pragma once

#include <ATen/detail/AcceleratorHooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

namespace at {

struct TORCH_API OpenCLHooksInterface : AcceleratorHooksInterface {
  ~OpenCLHooksInterface() override = default;

  void init() const override {
    // SYCL TODO: needs_review - initialize SYCL runtime
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    (void)device_index;
    return true;
  }

  DeviceIndex deviceCount() const {
    return 1; // SYCL TODO: needs_review - enumerate Intel GPU devices
  }

  DeviceIndex currentDevice() const {
    return 0;
  }

  bool isPinnedPtr(const void* data) const override {
    (void)data;
    return false; // SYCL TODO: needs_review
  }

  Allocator* getPinnedMemoryAllocator() const override {
    TORCH_CHECK(false, "OpenCL: pinned memory allocator not yet implemented");
  }

  const Generator& getDefaultGenerator(
      DeviceIndex device_index = -1) const {
    (void)device_index;
    TORCH_CHECK(false, "OpenCL: default generator not yet implemented");
  }

  Device getDeviceFromPtr(void* data) const override {
    (void)data;
    return Device(DeviceType::OPENCL, 0);
  }
};

struct TORCH_API OpenCLHooksArgs {};

TORCH_DECLARE_REGISTRY(OpenCLHooksRegistry, OpenCLHooksInterface, OpenCLHooksArgs);
#define REGISTER_OPENCL_HOOKS(clsname) \
  C10_REGISTER_CLASS(OpenCLHooksRegistry, clsname, clsname)

TORCH_API const OpenCLHooksInterface& getOpenCLHooks();

namespace detail {
TORCH_API const OpenCLHooksInterface& getOpenCLHooks();
} // namespace detail

} // namespace at
