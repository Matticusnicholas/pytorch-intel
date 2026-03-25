// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ValidateCompressedIndicesKernel.cu

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/ValidateCompressedIndicesCommon.h>
#include <ATen/native/sycl/Loops.h>

namespace at::native {

namespace {

template <typename func_t>
struct SYCLKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    sycl_kernel(iter, f);
  }
};

}

void _validate_compressed_sparse_indices_opencl(
    const bool is_crow,
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  validate_compressed_sparse_indices_kernel<SYCLKernelLauncher>(
      is_crow, cidx, idx, cdim, dim, nnz);
}

} // namespace at::native
