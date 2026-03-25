// AI-TRANSLATED: OpenCL device functions for Intel GPU (Arc/Xe) support
#include <c10/opencl/OpenCLFunctions.h>

// SYCL TODO: needs_review - Replace stubs with actual SYCL runtime calls
// #include <sycl/sycl.hpp>

namespace c10::opencl {

namespace {
thread_local DeviceIndex current_device_index = 0;
} // anonymous namespace

DeviceIndex device_count() {
  // SYCL TODO: needs_review - implement via sycl::device::get_devices(sycl::info::device_type::gpu)
  // For Intel GPUs, filter by sycl::info::device::vendor_id == 0x8086
  return 1; // Stub: assume 1 Intel GPU
}

DeviceIndex current_device() {
  return current_device_index;
}

void set_device(DeviceIndex device) {
  TORCH_CHECK(device >= 0 && device < device_count(),
    "OpenCL: invalid device index ", device);
  current_device_index = device;
}

bool is_available() {
  // SYCL TODO: needs_review - check for actual SYCL GPU devices
  // auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  // return !devices.empty();
#ifdef USE_OPENCL
  return true;
#else
  return false;
#endif
}

} // namespace c10::opencl
