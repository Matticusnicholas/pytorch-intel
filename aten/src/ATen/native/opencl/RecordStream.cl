// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/RecordStream.cu

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <c10/sycl/SYCLCachingAllocator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/record_stream_native.h>
#endif

namespace at::native {
void record_stream_cuda(Tensor& self, c10::Stream stream) {
  struct c10::StreamData3 data = stream.pack3();
  c10::sycl::SYCLCachingAllocator::recordStream(self.storage().data_ptr(), at::sycl::SYCLStream::unpack3(data.stream_id, data.device_index, data.device_type));
}
}  // namespace at::native
