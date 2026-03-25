// AI-TRANSLATED: Register OpenCL allocator with PyTorch
#include <ATen/Context.h>
#include <c10/core/Allocator.h>
#include <c10/opencl/OpenCLCachingAllocator.h>

namespace at::opencl {

static c10::opencl::OpenCLCachingAllocator g_opencl_alloc;

REGISTER_ALLOCATOR(c10::kOpenCL, &g_opencl_alloc)

} // namespace at::opencl
