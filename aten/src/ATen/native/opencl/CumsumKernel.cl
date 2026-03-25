// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/CumsumKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>

#include <ATen/native/sycl/ScanKernels.h>
#include <ATen/native/sycl/ScanUtils.h>

namespace at::native {

void launch_cumsum_opencl_kernel(const TensorBase& result, const TensorBase& self, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      self.scalar_type(), "cumsum_opencl",
      [&]() {
        scalar_t init = 0;
        scan_dim<scalar_t>(
            self,
            result,
            dim,
            init,
            std::plus<scalar_t>());
      });
}

} // namespace at::native
