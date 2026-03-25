// AI-TRANSLATED: OpenCL device functions for Intel GPU (Arc/Xe) support
#pragma once

#include <c10/core/Device.h>
#include <c10/opencl/OpenCLMacros.h>

namespace c10::opencl {

C10_OPENCL_API DeviceIndex device_count();
C10_OPENCL_API DeviceIndex current_device();
C10_OPENCL_API void set_device(DeviceIndex device);
C10_OPENCL_API bool is_available();

} // namespace c10::opencl
