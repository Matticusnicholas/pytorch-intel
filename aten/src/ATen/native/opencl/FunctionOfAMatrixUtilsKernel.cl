// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/FunctionOfAMatrixUtilsKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/FunctionOfAMatrixUtils.h>

#include <ATen/Dispatch.h>
#include <ATen/native/sycl/Loops.h>
#include <ATen/native/sycl/OffsetCalculator.h>
#include <ATen/sycl/Atomic.h>
#include <ATen/sycl/CUDAContext.h>

namespace at::native {

namespace {

template <int n_threads, int n_elems_per_thread, typename func_t>
C10_LAUNCH_BOUNDS_2(n_threads, n_elems_per_thread)
/* SYCL kernel */ void _elemwise_kernel(int total_n_elems, func_t f) {
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  int idx = total_work_block * item.get_group(0) + item.get_local_id(0);

  #pragma unroll
  for (int i = 0; i < n_elems_per_thread; ++i) {
    if (idx < total_n_elems) {
      f(idx);
      idx += n_threads;
    }
  }
}

template <int n_threads, int n_elems_per_thread, typename func_t>
void _lauch_kernel(int total_n_elems, const func_t& f) {
  TORCH_INTERNAL_ASSERT(
    total_n_elems >= 0 && total_n_elems <= std::numeric_limits<int32_t>::max()
  );

  sycl::range<3> block(n_threads);
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  sycl::range<3> grid((total_n_elems + total_work_block - 1) / total_work_block);

  auto stream = at::sycl::getCurrentSYCLStream();
  _elemwise_kernel<n_threads, n_elems_per_thread, func_t>
    /* SYCL: submit kernel with nd_range({grid=grid, block=block, smem=0, stream=stream}) */(total_n_elems, f);
  /* SYCL: kernel launch check handled by SYCL runtime */;
}

template <typename scalar_t>
void _compute_linear_combination_internal_kernel(
  TensorIterator& iter,
  int32_t in_stride,
  int32_t coeff_stride,
  int32_t num_summations
) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _compute_linear_combination_internal_kernel<scalar_t>(
        sub_iter, in_stride, coeff_stride, num_summations
      );
    }
    return;
  }

  auto offset_calc = make_offset_calculator<3>(iter);
  char* __restrict__ out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* __restrict__ in_ptr = reinterpret_cast<char*>(iter.data_ptr(1));
  char* __restrict__ coeff_ptr = reinterpret_cast<char*>(iter.data_ptr(2));

  auto loop = [=]C10_DEVICE(int idx) {
    auto offsets = offset_calc.get(idx);

    auto* __restrict__ out_data = reinterpret_cast<scalar_t*>(
      out_ptr + offsets[0]
    );
    auto* __restrict__ in_data = reinterpret_cast<scalar_t*>(
      in_ptr + offsets[1]
    );
    using primitive_t = typename scalar_value_type<scalar_t>::type;
    auto* __restrict__ coeff_data = reinterpret_cast<primitive_t*>(
      coeff_ptr + offsets[2]
    );

    // perform summation
    for (int32_t i = 0; i < num_summations; ++i) {
      *out_data += in_data[i * in_stride] * coeff_data[i * coeff_stride];
    }
  };

  _lauch_kernel<num_threads(), thread_work_size()>(iter.numel(), loop);
}

void _compute_linear_combination_opencl_kernel(
  TensorIterator& iter,
  int64_t in_stride,
  int64_t coeff_stride,
  int64_t num_summations
) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16,
    iter.dtype(),
    "_compute_linear_combination_opencl", [&] () {
      _compute_linear_combination_internal_kernel<scalar_t>(
        iter, in_stride, coeff_stride, num_summations
      );
    }
  );
}

}

REGISTER_DISPATCH(_compute_linear_combination_stub, &_compute_linear_combination_opencl_kernel)

} // namespace at::native
