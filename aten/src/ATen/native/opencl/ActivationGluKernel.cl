// AI-TRANSLATED: This file was automatically translated from CUDA to SYCL/OpenCL
// for Intel GPU (Arc/Xe) support. Requires hardware validation on Intel Arc GPU.
// Source: aten/src/ATen/native/cuda/ActivationGluKernel.cu
#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

// SYCL: thrust not available, using oneDPL or manual implementation

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/sycl/SYCLMathCompat.h>
#include <ATen/native/sycl/ApplyGridUtils.h>
#include <ATen/native/sycl/OffsetCalculator.h>
#include <ATen/native/sycl/Loops.h>

namespace at::native {

// -----------------------------------
// glu forward
// -----------------------------------
void glu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "glu_opencl", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        sycl_kernel(iter, [] SYCL_LAMBDA(scalar_t a_, scalar_t b_) -> scalar_t {
          const opmath_t a = a_;
          const opmath_t b = b_;
          const opmath_t one = opmath_t(1);
          const opmath_t sigmoid = one / (one + std::exp(-b));
          return a * sigmoid;
        });
      });
}

// -----------------------------------
// glu forward ad
// -----------------------------------
void glu_jvp_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "glu_opencl", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        sycl_kernel(
            iter,
            [] SYCL_LAMBDA(
                scalar_t res_, scalar_t b_, scalar_t da_, scalar_t db_)
                -> scalar_t {
              const opmath_t res = res_;
              const opmath_t b = b_;
              const opmath_t da = da_;
              const opmath_t db = db_;
              const opmath_t one = opmath_t(1);

              const opmath_t sig_b = one / (one + std::exp(-b));
              return (da * sig_b + res * (db - sig_b * db));
            });
      });
}

// -----------------------------------
// glu backward
// -----------------------------------

template <typename T>
T* byte_offset(T* ptr, int64_t offset) {
  using byte_ptr_t = typename std::
      conditional_t<std::is_const_v<T>, const char*, char*>;
  return reinterpret_cast<T*>(reinterpret_cast<byte_ptr_t>(ptr) + offset);
}

// SYCL TODO: needs_review - complex kernel requires manual optimization for Intel GPU
// The glu_backward_kernel was a __global__ CUDA kernel with <<<>>> launch syntax.
// Translated to a SYCL parallel_for pattern.
template <typename scalar_t, typename OffsetCalc>
void glu_backward_kernel_sycl(
    sycl::queue& queue,
    int numel,
    scalar_t* gI,
    const scalar_t* I,
    const scalar_t* gO,
    OffsetCalc offset_calculator,
    int64_t gI_byte_offset,
    int64_t I_byte_offset) {
  using opmath_t = at::opmath_type<scalar_t>;

  constexpr int64_t block_size = 256;
  const int64_t grid = (numel + block_size - 1) / block_size;

  queue.parallel_for(
      sycl::nd_range<1>(grid * block_size, block_size),
      [=](sycl::nd_item<1> item) {
        const uint32_t linear_index = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
        if (linear_index >= numel) {
          return;
        }
        const auto offsets = offset_calculator.get(linear_index);

        const opmath_t a = I[offsets[1]];
        const opmath_t b = *byte_offset(I + offsets[1], I_byte_offset);
        const opmath_t gO_val = gO[offsets[2]];

        const auto one = opmath_t(1);
        const opmath_t sigmoid = one / (one + std::exp(-b));

        auto* gA = gI + offsets[0];
        *gA = sigmoid * gO_val;

        auto* gB = byte_offset(gA, gI_byte_offset);
        *gB = (one - sigmoid) * sigmoid * gO_val * a;
      });
}

void launch_glu_backward_kernel(
    const TensorIteratorBase& iter,
    int64_t gI_stride,
    int64_t I_stride) {
  const auto N = iter.numel();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      N > 0 && N <= std::numeric_limits<int32_t>::max());
  const auto offset_calculator = make_element_offset_calculator<3>(iter);
  auto& queue = at::sycl::getCurrentSYCLQueue();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "glu_backward_opencl", [&] {
        auto gI = static_cast<scalar_t*>(iter.data_ptr(0));
        auto I = static_cast<const scalar_t*>(iter.data_ptr(1));
        auto gO = static_cast<const scalar_t*>(iter.data_ptr(2));
        glu_backward_kernel_sycl(
            queue,
            N,
            gI,
            I,
            gO,
            offset_calculator,
            gI_stride * sizeof(scalar_t),
            I_stride * sizeof(scalar_t));
      });
}

REGISTER_DISPATCH(glu_stub, &glu_kernel)
REGISTER_DISPATCH(glu_jvp_stub, &glu_jvp_kernel)

} // namespace at::native
