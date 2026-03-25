// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/FlattenIndicesKernel.cu
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/FlattenIndicesCommon.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/sycl/KernelUtils.h>
#include <ATen/native/sycl/OffsetCalculator.h>
#include <ATen/AccumulateType.h>

namespace at::native {

namespace {

template <typename func_t>
struct CUDAKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    sycl_kernel(iter, f);
  }
};

Tensor flatten_indices_opencl_kernel(const Tensor& indices, IntArrayRef size) {
  return _flatten_indices<CUDAKernelLauncher>(indices, size);
}

}

REGISTER_OPENCL_DISPATCH(flatten_indices_stub, &flatten_indices_opencl_kernel)

} // namespace at::native
