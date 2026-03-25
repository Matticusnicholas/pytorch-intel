// AI-TRANSLATED: OpenCL/SYCL stream management implementation
#include <c10/opencl/OpenCLStream.h>
#include <c10/util/Exception.h>

namespace c10::opencl {

namespace {
// SYCL TODO: needs_review - In production, manage actual sycl::queue pool
// For now, use a single default stream per device
thread_local StreamId current_stream_id = 0;
} // anonymous namespace

OpenCLStream getStreamFromPool(bool isHighPriority, DeviceIndex device_index) {
  (void)isHighPriority;
  if (device_index == -1) device_index = 0;
  return OpenCLStream(Stream(Stream::DEFAULT, Device(DeviceType::OPENCL, device_index)));
}

OpenCLStream getDefaultOpenCLStream(DeviceIndex device_index) {
  if (device_index == -1) device_index = 0;
  return OpenCLStream(Stream(Stream::DEFAULT, Device(DeviceType::OPENCL, device_index)));
}

OpenCLStream getCurrentOpenCLStream(DeviceIndex device_index) {
  if (device_index == -1) device_index = 0;
  return OpenCLStream(Stream(
      Stream::UNSAFE,
      Device(DeviceType::OPENCL, device_index),
      current_stream_id));
}

void setCurrentOpenCLStream(OpenCLStream stream) {
  current_stream_id = stream.id();
}

} // namespace c10::opencl
