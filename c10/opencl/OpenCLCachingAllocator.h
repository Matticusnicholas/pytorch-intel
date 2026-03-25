// AI-TRANSLATED: OpenCL caching allocator for Intel GPU (Arc/Xe) support
// Requires hardware validation on Intel Arc GPU.
// Modeled after c10/cuda/CUDACachingAllocator.h
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/opencl/OpenCLMacros.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace c10::opencl {

// Forward declaration for OpenCL memory handle
// SYCL TODO: needs_review - replace with actual sycl::queue and USM pointer types
using opencl_mem_t = void*;

C10_OPENCL_API void* opencl_malloc(size_t size, DeviceIndex device_index);
C10_OPENCL_API void opencl_free(void* ptr);

// Simple block-based caching allocator for OpenCL/SYCL unified shared memory
class C10_OPENCL_API OpenCLCachingAllocator final : public Allocator {
 public:
  OpenCLCachingAllocator();
  ~OpenCLCachingAllocator() override;

  DataPtr allocate(size_t nbytes) override;
  DeleterFnPtr raw_deleter() const override;

  void emptyCache();
  size_t totalAllocatedMemory() const;
  size_t totalCachedMemory() const;

 private:
  struct Block {
    void* ptr;
    size_t size;
    DeviceIndex device;
    bool in_use;
  };

  void* malloc_impl(size_t size, DeviceIndex device);
  void free_impl(void* ptr);

  mutable std::mutex mutex_;
  std::unordered_map<void*, Block> allocated_blocks_;
  std::vector<Block> free_blocks_;
  size_t total_allocated_ = 0;
  size_t total_cached_ = 0;
};

C10_OPENCL_API OpenCLCachingAllocator* getOpenCLCachingAllocator();

} // namespace c10::opencl
