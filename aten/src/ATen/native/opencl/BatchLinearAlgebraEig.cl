// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/BatchLinearAlgebraEig.cu
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <c10/sycl/SYCLGuard.h>

namespace at::native {

namespace {

template <typename scalar_t>
void linalg_eig_make_complex_eigenvectors_sycl_impl(
    const Tensor& complex_vectors,
    const Tensor& complex_values,
    const Tensor& real_vectors) {

  const auto n = real_vectors.size(-1);
  const auto matrix_stride = matrixStride(real_vectors);
  const auto batch_size = batchCount(real_vectors);

  if (batch_size == 0 || n == 0) return;

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_vectors.mT().is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(complex_values.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(real_vectors.mT().is_contiguous());

  const int64_t total_elements = batch_size * n * n;
  const int threads = 256;
  const int blocks = (total_elements + threads - 1) / threads;

  auto* result_ptr = complex_vectors.data_ptr<c10::complex<scalar_t>>();
  const auto* eigenvalues_ptr = complex_values.const_data_ptr<c10::complex<scalar_t>>();
  const auto* vectors_ptr = real_vectors.const_data_ptr<scalar_t>();

  auto& queue = at::sycl::getCurrentSYCLQueue();
  queue.parallel_for(
      sycl::nd_range<1>(blocks * threads, threads),
      [=](sycl::nd_item<1> item) {
        const int64_t idx = item.get_global_id(0);
        if (idx >= total_elements) return;

        const int64_t batch_idx = idx / (n * n);
        const int64_t local_idx = idx % (n * n);
        const int64_t col = local_idx / n;
        const int64_t row = local_idx % n;

        const auto* batch_eigenvalues = eigenvalues_ptr + batch_idx * n;
        const auto* batch_vectors = vectors_ptr + batch_idx * matrix_stride;
        auto* batch_result = result_ptr + batch_idx * matrix_stride;

        const auto eigenvalue = batch_eigenvalues[col];

        if (eigenvalue.imag() == scalar_t(0)) {
          batch_result[col * n + row] = c10::complex<scalar_t>(
              batch_vectors[col * n + row], scalar_t(0));
        } else if (eigenvalue.imag() > scalar_t(0)) {
          batch_result[col * n + row] = c10::complex<scalar_t>(
              batch_vectors[col * n + row],
              batch_vectors[(col + 1) * n + row]);
        } else {
          batch_result[col * n + row] = c10::complex<scalar_t>(
              batch_vectors[(col - 1) * n + row],
              -batch_vectors[col * n + row]);
        }
      });
}

void linalg_eig_make_complex_eigenvectors_opencl(
    const Tensor& complex_vectors,
    const Tensor& complex_values,
    const Tensor& real_vectors) {

  TORCH_INTERNAL_ASSERT(complex_vectors.is_cuda());
  TORCH_INTERNAL_ASSERT(complex_values.is_cuda());
  TORCH_INTERNAL_ASSERT(real_vectors.is_cuda());

  c10::sycl::SYCLGuard device_guard(real_vectors.device());

  AT_DISPATCH_V2(
      real_vectors.scalar_type(),
      "linalg_eig_make_complex_eigenvectors_opencl",
      AT_WRAP([&] {
        linalg_eig_make_complex_eigenvectors_sycl_impl<scalar_t>(
            complex_vectors, complex_values, real_vectors);
      }),
      AT_EXPAND(AT_FLOATING_TYPES));
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(linalg_eig_make_complex_eigenvectors_stub, &linalg_eig_make_complex_eigenvectors_opencl)

} // namespace at::native
