// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/SpectralOps.cpp
//
// SYCL TODO: needs_review - replace with oneMKL FFT equivalent
// The CUDA version uses cuFFT for FFT operations. For the SYCL/OpenCL backend,
// these should use oneMKL DFT (oneapi::mkl::dft) equivalents.

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/sycl/SYCLContext.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/detail/OpenCLHooksInterface.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fft_c2c_native.h>
#include <ATen/ops/_fft_c2r_native.h>
#include <ATen/ops/_fft_r2c_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mul.h>
#endif

#include <cmath>

namespace at::native {

using namespace at::native::detail;

// SYCL TODO: needs_review - replace with oneMKL FFT equivalent
// The oneMKL DFT descriptor replaces CuFFTConfig/CuFFTParams.
// Use oneapi::mkl::dft::descriptor for plan creation and execution.

namespace {
constexpr int64_t mklfft_max_ndim = 3;

bool has_large_prime_factor(int64_t n) {
  constexpr int64_t first_large_prime = 11;
  const std::array<int64_t, 4> prime_radices{{2, 3, 5, 7}};
  for (auto prime : prime_radices) {
    if (n < first_large_prime) {
        return false;
    }

    while (n % prime == 0) {
      n /= prime;
    }
  }
  return n != 1;
}

// SYCL TODO: needs_review - replace with oneMKL FFT equivalent
// This function wraps the FFT execution. For oneMKL, use
// oneapi::mkl::dft::compute_forward / compute_backward.
const Tensor& _exec_fft(Tensor& out, const Tensor& self, IntArrayRef out_sizes,
                         IntArrayRef dim, bool forward) {
  const auto ndim = self.dim();
  const int64_t signal_ndim = dim.size();
  const auto batch_dims = ndim - signal_ndim;

  DimVector dim_permute(ndim);
  std::iota(dim_permute.begin(), dim_permute.end(), int64_t{0});

  c10::SmallVector<bool, kDimVectorStaticSize> is_transformed_dim(ndim);
  for (const auto& d : dim) {
    is_transformed_dim[d] = true;
  }
  auto batch_end = std::partition(dim_permute.begin(), dim_permute.end(),
                                  [&](int64_t d) {return !is_transformed_dim[d]; });
  auto self_strides = self.strides();
  std::sort(dim_permute.begin(), batch_end,
            [&](int64_t a, int64_t b) { return self_strides[a] > self_strides[b]; });
  std::copy(dim.cbegin(), dim.cend(), batch_end);
  auto input = self.permute(dim_permute);

  DimVector batched_sizes(signal_ndim + 1);
  batched_sizes[0] = -1;
  std::copy(input.sizes().cbegin() + batch_dims, input.sizes().cend(), batched_sizes.begin() + 1);
  input = input.reshape(batched_sizes);

  const auto batch_size = input.sizes()[0];
  DimVector signal_size(signal_ndim + 1);
  signal_size[0] = batch_size;
  for (const auto i : c10::irange(signal_ndim)) {
    auto in_size = input.sizes()[i + 1];
    auto out_size = out_sizes[dim[i]];
    signal_size[i + 1] = std::max(in_size, out_size);
    TORCH_INTERNAL_ASSERT(in_size == signal_size[i + 1] ||
                          in_size == (signal_size[i + 1] / 2) + 1);
    TORCH_INTERNAL_ASSERT(out_size == signal_size[i + 1] ||
                          out_size == (signal_size[i + 1] / 2) + 1);
  }

  batched_sizes[0] = batch_size;
  DimVector batched_out_sizes(batched_sizes.begin(), batched_sizes.end());
  for (const auto i : c10::irange(dim.size())) {
    batched_out_sizes[i + 1] = out_sizes[dim[i]];
  }
  out.resize_(batched_out_sizes, MemoryFormat::Contiguous);

  // SYCL TODO: needs_review - replace with oneMKL FFT equivalent
  // Create oneMKL DFT descriptor and execute the transform.
  // For now, this is a structural placeholder that mirrors the cuFFT flow.
  const auto value_type = c10::toRealValueType(input.scalar_type());

  // Execute the FFT using oneMKL DFT
  // oneapi::mkl::dft::descriptor<precision, domain> desc(signal_size);
  // desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, DFTI_NOT_INPLACE);
  // desc.commit(at::opencl::getCurrentOpenCLStream().queue());
  // oneapi::mkl::dft::compute_forward(desc, input_data, out_data);

  TORCH_CHECK(false, "OpenCL FFT backend not yet implemented. "
              "Requires oneMKL DFT integration.");

  // Inplace reshaping to original batch shape and inverting the dimension permutation
  DimVector out_strides(ndim);
  int64_t batch_numel = 1;
  for (int64_t i = batch_dims - 1; i >= 0; --i) {
    out_strides[dim_permute[i]] = batch_numel * out.strides()[0];
    batch_numel *= out_sizes[dim_permute[i]];
  }
  for (const auto i : c10::irange(batch_dims, ndim)) {
    out_strides[dim_permute[i]] = out.strides()[1 + (i - batch_dims)];
  }
  return out.as_strided_(out_sizes, out_strides, out.storage_offset());
}

double _fft_normalization_scale(int64_t normalization, IntArrayRef sizes, IntArrayRef dims) {
  auto norm = static_cast<fft_norm_mode>(normalization);
  if (norm == fft_norm_mode::none) {
    return 1.0;
  }

  int64_t signal_numel = 1;
  for (auto dim : dims) {
    signal_numel *= sizes[dim];
  }
  const double scale_denom = (norm == fft_norm_mode::by_root_n) ?
    std::sqrt(signal_numel) : static_cast<double>(signal_numel);
  return 1.0 / scale_denom;
}

const Tensor& _fft_apply_normalization(const Tensor& self, int64_t normalization, IntArrayRef sizes, IntArrayRef dims) {
  auto scale = _fft_normalization_scale(normalization, sizes, dims);
  return (scale == 1.0) ? self : self.mul_(scale);
}

Tensor& _fft_apply_normalization_out(Tensor& out, const Tensor& self, int64_t normalization, IntArrayRef sizes, IntArrayRef dims) {
  auto scale = _fft_normalization_scale(normalization, sizes, dims);
  return at::mul_out(out, self, c10::scalar_to_tensor(scale));
}

}  // namespace (anonymous)

bool use_optimized_mklfft_path(IntArrayRef dim) {
  if (dim.size() > mklfft_max_ndim || (
    dim.size() >= 2 && dim[0] == 0 && dim[1] == 1
  )) {
    return false;
  } else {
    return true;
  }
}

// n-dimensional real to complex FFT
Tensor _fft_r2c_opencl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool onesided) {
  TORCH_CHECK(self.is_floating_point());
  // SYCL TODO: needs_review - replace with oneMKL FFT equivalent
  TORCH_CHECK(false, "OpenCL _fft_r2c not yet implemented. Requires oneMKL DFT integration.");
  return self;
}

Tensor& _fft_r2c_opencl_out(const Tensor& self, IntArrayRef dim,
                           int64_t normalization, bool onesided, Tensor& out) {
  // SYCL TODO: needs_review - replace with oneMKL FFT equivalent
  TORCH_CHECK(false, "OpenCL _fft_r2c_out not yet implemented. Requires oneMKL DFT integration.");
  return out;
}

// n-dimensional complex to real IFFT
Tensor _fft_c2r_opencl(const Tensor& self, IntArrayRef dim, int64_t normalization, int64_t lastdim) {
  TORCH_CHECK(self.is_complex());
  // SYCL TODO: needs_review - replace with oneMKL FFT equivalent
  TORCH_CHECK(false, "OpenCL _fft_c2r not yet implemented. Requires oneMKL DFT integration.");
  return self;
}

Tensor& _fft_c2r_opencl_out(const Tensor& self, IntArrayRef dim,
                           int64_t normalization, int64_t lastdim, Tensor& out) {
  // SYCL TODO: needs_review - replace with oneMKL FFT equivalent
  TORCH_CHECK(false, "OpenCL _fft_c2r_out not yet implemented. Requires oneMKL DFT integration.");
  return out;
}

// n-dimensional complex to complex FFT/IFFT
Tensor _fft_c2c_opencl(const Tensor& self, IntArrayRef dim, int64_t normalization, bool forward) {
  TORCH_CHECK(self.is_complex());
  if (dim.empty()) {
    return self.clone();
  }
  // SYCL TODO: needs_review - replace with oneMKL FFT equivalent
  TORCH_CHECK(false, "OpenCL _fft_c2c not yet implemented. Requires oneMKL DFT integration.");
  return self;
}

Tensor& _fft_c2c_opencl_out(const Tensor& self, IntArrayRef dim,
                           int64_t normalization, bool forward, Tensor& out) {
  // SYCL TODO: needs_review - replace with oneMKL FFT equivalent
  TORCH_CHECK(false, "OpenCL _fft_c2c_out not yet implemented. Requires oneMKL DFT integration.");
  return out;
}

} // at::native
